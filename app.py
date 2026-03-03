import os
import uuid
import threading
import logging
import subprocess
import shutil
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — use system env vars

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, redirect, url_for,
)
from werkzeug.utils import secure_filename

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── App config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
_base = os.path.dirname(os.path.abspath(__file__))
app.config['SECRET_KEY']       = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
app.config['UPLOAD_FOLDER']    = os.environ.get('UPLOAD_FOLDER',  os.path.join(_base, 'uploads'))
app.config['OUTPUT_FOLDER']    = os.environ.get('OUTPUT_FOLDER',  os.path.join(_base, 'outputs'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024  # 16 GB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# In-memory task store (keyed by task_id)
tasks: dict = {}
TASKS_LOCK = threading.Lock()

# ── Allowed file types ────────────────────────────────────────────────────────
ALLOWED_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv'}
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTS


def _is_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTS


# ── Locate Meshroom ───────────────────────────────────────────────────────────
def find_meshroom() -> str | None:
    # Check explicit env var first
    env_bin = os.environ.get('MESHROOM_BINARY', '')
    if env_bin and os.path.isfile(env_bin):
        return env_bin

    # Well-known Linux paths
    candidates = [
        str(Path.home() / 'Meshroom-2023.3.0' / 'meshroom_batch'),
        str(Path.home() / 'Meshroom' / 'meshroom_batch'),
        '/opt/Meshroom/meshroom_batch',
    ]
    # Glob for any Meshroom install under home
    candidates += [str(p) for p in Path.home().glob('Meshroom*/meshroom_batch')]

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fall back to PATH
    return shutil.which('meshroom_batch')


MESHROOM_BIN = find_meshroom()
if MESHROOM_BIN:
    log.info('Meshroom found at: %s', MESHROOM_BIN)
else:
    log.warning('Meshroom not found! Set MESHROOM_BINARY env var.')


def _meshroom_env() -> dict:
    """Build env that lets Meshroom find its bundled CUDA 11 libs on Linux."""
    env = os.environ.copy()
    if MESHROOM_BIN:
        meshroom_root = Path(MESHROOM_BIN).parent
        extra = [
            str(meshroom_root / 'aliceVision' / 'lib'),
            str(meshroom_root / 'lib'),
        ]
        existing = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = ':'.join(extra + ([existing] if existing else []))
        env['ALICEVISION_ROOT'] = str(meshroom_root / 'aliceVision')
    return env


# ── Pipeline stages → progress % ────────────────────────────────────────────
# Mapped both from aliceVision binary names (real-time process detection)
# AND from stdout keywords (fallback)
STAGE_PROGRESS = {
    'CameraInit':          8,
    'FeatureExtraction':  18,
    'ImageMatching':      28,
    'FeatureMatching':    38,
    'StructureFromMotion': 50,
    'PrepareDenseScene':  60,
    'DepthMap':           70,
    'DepthMapFilter':     76,
    'Meshing':            83,
    'MeshFiltering':      88,
    'Texturing':          94,
}

# Maps partial aliceVision binary names → progress % (for process-based detection)
PROCESS_STAGE_MAP = [
    ('cameraInit',        'CameraInit',          8),
    ('featureExtraction', 'FeatureExtraction',   18),
    ('imageMatching',     'ImageMatching',       28),
    ('featureMatching',   'FeatureMatching',     38),
    ('structureFromMotion','StructureFromMotion', 50),
    ('prepareDenseScene', 'PrepareDenseScene',   60),
    ('depthMapEstimation','DepthMap',            70),
    ('depthMapFilter',    'DepthMapFilter',      76),
    ('meshing',           'Meshing',             83),
    ('meshFiltering',     'MeshFiltering',       88),
    ('texturing',         'Texturing',           94),
]


def _watch_meshroom_progress(task: dict, proc: 'subprocess.Popen[str]', stop_evt: threading.Event) -> None:
    """Poll running aliceVision processes every 3s and update task progress."""
    import re
    while not stop_evt.is_set():
        try:
            result = subprocess.run(
                ['pgrep', '-a', 'aliceVision'],
                capture_output=True, text=True,
            )
            procs = result.stdout.lower()
            for key, stage, pct in PROCESS_STAGE_MAP:
                if key.lower() in procs:
                    if task.get('progress', 0) < pct:
                        task['progress'] = pct
                        task['status'] = stage
                        log.info('Stage detected via process: %s (%d%%)', stage, pct)
                    break
        except Exception:
            pass
        stop_evt.wait(3)



# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(task_id: str) -> None:
    task = tasks.get(task_id)
    if not task:
        return

    task_dir   = task['path']
    input_type = task['input_type']
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], task_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        images_dir = task_dir

        # ── Step 1: Extract frames (video only) ───────────────────────────────
        if input_type == 'video':
            task['status'] = 'extracting'
            task['progress'] = 5
            video_file = next((f for f in os.listdir(task_dir) if _is_video(f)), None)
            if not video_file:
                raise FileNotFoundError('No video file found in upload directory')

            frames_dir = os.path.join(task_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            fps = os.environ.get('FFMPEG_FPS', '2')
            subprocess.run(
                ['ffmpeg', '-i', os.path.join(task_dir, video_file),
                 '-vf', f'fps={fps}',
                 os.path.join(frames_dir, 'frame_%04d.jpg')],
                check=True, capture_output=True,
            )
            images_dir = frames_dir
            log.info('[%s] Extracted frames to %s', task_id, frames_dir)

        # Count input images
        n_images = sum(1 for f in os.listdir(images_dir) if _is_image(f))
        log.info('[%s] Found %d images', task_id, n_images)
        if n_images < 3:
            raise RuntimeError(f'Need at least 3 images, got {n_images}')

        # ── Step 2: Meshroom reconstruction ───────────────────────────────────
        task['status'] = 'reconstructing'
        task['progress'] = 8

        if not MESHROOM_BIN:
            raise FileNotFoundError(
                'meshroom_batch not found. Install Meshroom and set MESHROOM_BINARY.'
            )

        cmd = [
            MESHROOM_BIN,
            '--input',  images_dir,
            '--output', output_dir,
        ]
        log.info('[%s] Running: %s', task_id, ' '.join(cmd))

        env = _meshroom_env()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        task['proc'] = proc  # store so /cancel can kill it

        # Start process-watcher thread for real-time stage detection
        stop_watch = threading.Event()
        watcher = threading.Thread(
            target=_watch_meshroom_progress,
            args=(task, proc, stop_watch),
            daemon=True,
        )
        watcher.start()

        # Drain stdout (captures errors; stage progress via watcher instead)
        for line in proc.stdout:
            pass

        stop_watch.set()
        proc.wait()

        if task.get('status') == 'cancelled':
            return  # stopped by user

        if proc.returncode != 0:
            raise RuntimeError(f'Meshroom exited with code {proc.returncode}')

        # ── Step 3: Find output OBJ ───────────────────────────────────────────
        task['status'] = 'converting textures'
        task['progress'] = 95

        obj_files = list(Path(output_dir).rglob('texturedMesh.obj'))
        if not obj_files:
            obj_files = list(Path(output_dir).rglob('*.obj'))
        if not obj_files:
            raise FileNotFoundError('Meshroom did not produce an OBJ file')
        obj_path = max(obj_files, key=lambda p: p.stat().st_mtime)
        log.info('[%s] Using OBJ: %s', task_id, obj_path)

        # ── Step 4: EXR → PNG (so trimesh & model-viewer see real colours) ────
        obj_dir = obj_path.parent
        mtl_path = obj_path.with_suffix('.mtl')
        for exr_file in obj_dir.glob('*.exr'):
            png_path = exr_file.with_suffix('.png')
            if not png_path.exists():
                subprocess.run(
                    ['ffmpeg', '-i', str(exr_file), '-update', '1',
                     '-pix_fmt', 'rgb24', str(png_path), '-y'],
                    capture_output=True,
                )
                log.info('[%s] Converted %s → %s', task_id, exr_file.name, png_path.name)
            # Patch the MTL to reference the PNG instead of EXR
            if mtl_path.exists():
                mtl_text = mtl_path.read_text()
                if exr_file.name in mtl_text:
                    mtl_path.write_text(mtl_text.replace(exr_file.name, png_path.name))

        # ── Step 5: OBJ → GLB (for model-viewer) ─────────────────────────────
        task['status'] = 'exporting GLB'
        import trimesh
        mesh = trimesh.load(str(obj_path), process=False)
        if hasattr(mesh, 'is_empty') and mesh.is_empty:
            raise RuntimeError('Loaded mesh is empty')

        glb_filename = f'{task_id}_model.glb'
        glb_path = os.path.join(app.config['OUTPUT_FOLDER'], glb_filename)
        mesh.export(glb_path)
        log.info('[%s] Exported GLB: %s', task_id, glb_path)

        task['result_file'] = glb_filename
        task['status'] = 'complete'
        task['progress'] = 100

    except Exception as exc:
        log.error('[%s] Pipeline failed: %s', task_id, exc, exc_info=True)
        task['status'] = 'failed'
        task['error'] = str(exc)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files provided'}), 400

    task_id  = str(uuid.uuid4())
    task_dir = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
    os.makedirs(task_dir, exist_ok=True)

    input_type = 'images'

    if len(files) == 1 and _is_video(files[0].filename):
        input_type = 'video'
        f = files[0]
        f.save(os.path.join(task_dir, secure_filename(f.filename)))
    else:
        saved = 0
        for f in files:
            if not f or not f.filename:
                continue
            raw = f.filename.replace('\\', '/').split('/')[-1]
            fname = secure_filename(raw)
            if not fname or not _is_image(fname):
                continue
            f.save(os.path.join(task_dir, fname))
            saved += 1
        if saved == 0:
            shutil.rmtree(task_dir, ignore_errors=True)
            return jsonify({'error': 'No valid image files found'}), 400

    with TASKS_LOCK:
        tasks[task_id] = {
            'status':     'queued',
            'progress':   0,
            'input_type': input_type,
            'path':       task_dir,
        }

    threading.Thread(target=run_pipeline, args=(task_id,), daemon=True).start()
    return jsonify({'task_id': task_id})


@app.route('/processing/<task_id>')
def processing(task_id):
    if task_id not in tasks:
        return redirect(url_for('index'))
    return render_template('processing.html', task_id=task_id)


@app.route('/status/<task_id>')
def status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404

    response = dict(task)
    # Lazily attach thumbnail list (images directly in upload dir)
    if 'images' not in task:
        try:
            task['images'] = [f for f in os.listdir(task['path']) if _is_image(f)]
        except Exception:
            task['images'] = []
    response['images'] = task['images']
    # Don't serialize the proc object
    response.pop('proc', None)
    return jsonify(response)


@app.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    proc = task.get('proc')
    if proc and proc.poll() is None:
        try:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        task['status'] = 'cancelled'
        task['error']  = 'Stopped by user'
        log.info('[%s] Task cancelled by user', task_id)
    return jsonify({'status': task['status']})


@app.route('/uploads/<task_id>/<filename>')
def serve_upload(task_id, filename):
    if task_id not in tasks:
        return 'Not found', 404
    return send_from_directory(
        os.path.join(app.config['UPLOAD_FOLDER'], task_id), filename
    )


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


@app.route('/result/<task_id>')
def result(task_id):
    return redirect(url_for('processing', task_id=task_id))


if __name__ == '__main__':
    log.info('Starting 3D Reconstructor on http://0.0.0.0:5000')
    app.run(debug=False, use_reloader=False, port=5000, host='0.0.0.0')
