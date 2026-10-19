"""Resumable, deterministic diffuse render gallery; no source data is modified."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image

ANGLES = tuple((az, el) for el in (25, 45, 65, 80) for az in (0, 90, 180, 270))
SIZE = 192


def camera_pose(azimuth: float, elevation: float, distance: float = 2.0) -> np.ndarray:
    az, el = np.deg2rad([azimuth, elevation])
    position = np.array([distance * np.cos(el) * np.sin(az),
                         distance * np.sin(el),
                         distance * np.cos(el) * np.cos(az)])
    forward = position / np.linalg.norm(position)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, up, forward))
    pose[:3, 3] = position
    return pose


def normalized_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Invalid mesh")
    extent = float(np.max(mesh.extents))
    if extent <= 0:
        raise ValueError("Degenerate mesh")
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_scale(1.0 / extent)
    return mesh


def render_one(renderer: pyrender.OffscreenRenderer, path: Path, output: Path) -> dict:
    mesh = normalized_mesh(path)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.7, 0.7, 0.7, 1.0),
        metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
    scene = pyrender.Scene(bg_color=np.array([0., 0., 0., 1.]), ambient_light=np.array([0.25, 0.25, 0.25]))
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
    camera = pyrender.OrthographicCamera(xmag=0.85, ymag=0.85)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    output.mkdir(parents=True, exist_ok=True)
    produced = []
    for index, (az, el) in enumerate(ANGLES):
        pose = camera_pose(az, el)
        camera_node = scene.add(camera, pose=pose)
        light_node = scene.add(light, pose=pose)
        color, _ = renderer.render(scene)
        scene.remove_node(camera_node)
        scene.remove_node(light_node)
        image = Image.fromarray(color[:, :, :3].astype(np.uint8), mode="RGB")
        dest = output / f"view_{index:02d}.png"
        image.save(dest)
        produced.append(dest.name)
    return {"model_id": path.stem, "views": produced, "vertices": int(len(mesh.vertices))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parent / "protocol.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "cache" / "renders")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test first N CADs; omit for the full gallery")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    ids = protocol["gallery_model_ids"][:args.limit]
    renderer = pyrender.OffscreenRenderer(SIZE, SIZE)
    failures = []
    start = time.perf_counter()
    try:
        for index, model_id in enumerate(ids, 1):
            output = args.output / model_id
            if all((output / f"view_{v:02d}.png").is_file() for v in range(len(ANGLES))):
                continue
            try:
                render_one(renderer, args.root / "dataset" / "models" / f"{model_id}.stl", output)
            except Exception as error:
                failures.append({"model_id": model_id, "error": type(error).__name__ + ": " + str(error)[:120]})
            if index % 100 == 0 or index == len(ids):
                print(f"{index}/{len(ids)} models, {len(failures)} failures, {time.perf_counter()-start:.1f}s", flush=True)
    finally:
        renderer.delete()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "render_failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"{len(failures)} CAD render failures; see render_failures.json")


if __name__ == "__main__":
    main()
