"""Photogrammetry-style 3D reconstruction from video frames or image sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


@dataclass
class ReconstructionConfig:
    frame_step: int = 7
    max_features: int = 3500
    match_ratio: float = 0.78
    min_inliers: int = 45
    voxel_size: float = 0.02
    poisson_depth: int = 8


def extract_frames(video_path: str | Path, output_dir: str | Path, frame_step: int = 7) -> list[Path]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved_paths: list[Path] = []
    frame_index = 0
    saved_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            path = output_dir / f"frame_{saved_index:05d}.jpg"
            cv2.imwrite(str(path), frame)
            saved_paths.append(path)
            saved_index += 1
        frame_index += 1

    capture.release()
    return saved_paths


def _load_images(image_paths: Iterable[str | Path]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        max_dim = max(img.shape[:2])
        if max_dim > 1400:
            scale = 1400.0 / float(max_dim)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        images.append(img)
    return images


def _camera_matrix(width: int, height: int, fov_degrees: float = 60.0) -> np.ndarray:
    f = 0.5 * width / np.tan(np.deg2rad(fov_degrees) * 0.5)
    return np.array(
        [
            [f, 0.0, width / 2.0],
            [0.0, f, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _match_orb(
    img1: np.ndarray,
    img2: np.ndarray,
    max_features: int,
    match_ratio: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    if des1 is None or des2 is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < match_ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 20:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    return pts1, pts2


def _triangulate_pair(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    min_inliers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    if E is None or mask is None:
        return None

    inliers = mask.ravel().astype(bool)
    if inliers.sum() < min_inliers:
        return None

    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    _, R, t, pose_mask = cv2.recoverPose(E, pts1_in, pts2_in, K)
    pose_inliers = pose_mask.ravel().astype(bool)

    if pose_inliers.sum() < min_inliers:
        return None

    pts1_pose = pts1_in[pose_inliers]
    pts2_pose = pts2_in[pose_inliers]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pts4d = cv2.triangulatePoints(P1, P2, pts1_pose.T, pts2_pose.T)
    pts3d_c1 = (pts4d[:3] / (pts4d[3] + 1e-10)).T

    pts3d_c2 = (R @ pts3d_c1.T + t).T
    valid = np.isfinite(pts3d_c1).all(axis=1) & (pts3d_c1[:, 2] > 0) & (pts3d_c2[:, 2] > 0)

    if valid.sum() < 20:
        return None

    return pts3d_c1[valid], R, t


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(points) == 0:
        return points
    quantized = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(quantized, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def _statistical_outlier_filter(points: np.ndarray, k: int = 16, z_thresh: float = 2.2) -> np.ndarray:
    if len(points) < k + 1:
        return points
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_neighbor_dist = dists[:, 1:].mean(axis=1)
    mean = float(mean_neighbor_dist.mean())
    std = float(mean_neighbor_dist.std()) + 1e-9
    z = np.abs((mean_neighbor_dist - mean) / std)
    keep = z < z_thresh
    filtered = points[keep]
    return filtered if len(filtered) >= 50 else points


def _mesh_from_cloud(points: np.ndarray) -> trimesh.Trimesh:
    if len(points) < 4:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
    cloud = trimesh.points.PointCloud(points)
    mesh = cloud.convex_hull
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
    return mesh


def _write_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def reconstruct_3d(
    image_paths: list[str | Path],
    output_dir: str | Path,
    config: ReconstructionConfig,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(0.05, "Loading images")

    images = _load_images(image_paths)
    if len(images) < 2:
        raise RuntimeError("Need at least 2 valid images to build a 3D model")

    h, w = images[0].shape[:2]
    K = _camera_matrix(w, h)

    if progress:
        progress(0.15, "Estimating camera motion and triangulating points")

    world_points = []
    T_w_c = np.eye(4, dtype=np.float64)

    total_pairs = len(images) - 1
    for idx in range(total_pairs):
        pair = _match_orb(images[idx], images[idx + 1], config.max_features, config.match_ratio)
        if pair is None:
            continue

        pts1, pts2 = pair
        tri = _triangulate_pair(pts1, pts2, K, config.min_inliers)
        if tri is None:
            continue

        pts3d_c1, R, t = tri

        pts_h = np.hstack([pts3d_c1, np.ones((pts3d_c1.shape[0], 1))])
        pts_world = (T_w_c @ pts_h.T).T[:, :3]
        world_points.append(pts_world)

        T_c2_c1 = np.eye(4)
        T_c2_c1[:3, :3] = R
        T_c2_c1[:3, 3] = t.ravel()
        T_w_c = T_w_c @ np.linalg.inv(T_c2_c1)

        if progress:
            p = 0.15 + 0.45 * ((idx + 1) / max(1, total_pairs))
            progress(p, f"Processed image pair {idx + 1}/{total_pairs}")

    if not world_points:
        raise RuntimeError(
            "Could not triangulate enough stable points. "
            "Try clearer, overlapping frames with better lighting."
        )

    points = np.vstack(world_points)

    if progress:
        progress(0.68, "Cleaning point cloud")

    points = _voxel_downsample(points, config.voxel_size)
    points = _statistical_outlier_filter(points)

    if len(points) < 50:
        raise RuntimeError(
            "Point cloud is too sparse for meshing. "
            "Use more overlap and texture-rich images."
        )

    if progress:
        progress(0.82, "Building mesh")

    mesh = _mesh_from_cloud(points)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(
            "Mesh generation failed. "
            "Input appears too noisy or lacks camera motion diversity."
        )

    mesh_path = output_dir / "model.obj"
    cloud_path = output_dir / "point_cloud.ply"

    mesh.export(mesh_path)
    _write_point_cloud_ply(cloud_path, points)

    if progress:
        progress(1.0, "3D model complete")

    return {
        "mesh": str(mesh_path),
        "point_cloud": str(cloud_path),
        "points": str(len(points)),
        "faces": str(len(mesh.faces)),
    }
