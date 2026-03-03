# 3D Reconstructor

A web application that turns photos or videos into textured 3D models using [Meshroom](https://alicevision.org/#meshroom) (AliceVision).

## Features

- **Upload images or video** — drag-and-drop folder or file
- **Meshroom pipeline** — full SfM → depth maps → meshing → texturing
- **Live dashboard** — Meshroom Task Manager style progress table with animated progress bars
- **3D preview** — view the model in-browser via `<model-viewer>`
- **Download** — export as `.glb`
- **Stop button** — cancel a running job gracefully
- **Resizable panels** — drag the gallery width and task-manager height

## Requirements

- Python 3.10+
- [Meshroom 2023.3.0](https://github.com/alicevision/Meshroom/releases) (or set `MESHROOM_BINARY`)
- FFmpeg (for video input)
- NVIDIA GPU with CUDA 11+ (optional but strongly recommended)

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd "IIT Project"

# 2. Create virtualenv
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env if Meshroom is not in the default location

# 5. Run
python app.py
# Open http://localhost:5000
```

## Environment Variables

See [`.env.example`](.env.example) for all options.

## Project Structure

```
├── app.py              # Flask backend
├── src/
│   └── reconstruct.py  # (legacy — unused in current pipeline)
├── templates/
│   ├── base.html       # Shared head/fonts
│   ├── index.html      # Upload page
│   └── processing.html # Dashboard page
├── static/css/
│   ├── style.css       # Upload page styles
│   └── dashboard.css   # Dashboard styles
├── requirements.txt
├── .env.example
└── .gitignore
```

## Notes

- `uploads/`, `outputs/`, and `venv/` are gitignored — not committed
- The `.env` file contains secrets — never commit it
