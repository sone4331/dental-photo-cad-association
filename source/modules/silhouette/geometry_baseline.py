"""Evaluate the frozen silhouette/SDF baseline over all retrospective folds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def descriptor(path: Path, roll: int, size: int = 32) -> np.ndarray:
    """Same silhouette SDF as the archived shape_templates.py baseline."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((gray > 25).astype(np.uint8))
    if n < 2:
        raise ValueError(f"Empty foreground: {path}")
    mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    y, x = np.where(mask)
    cx, cy = x.mean(), y.mean()
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).max()
    mat = cv2.getRotationMatrix2D((float(cx), float(cy)), roll,
                                  .43 * size / max(radius, 1))
    mat[0, 2] += (size - 1) / 2 - cx
    mat[1, 2] += (size - 1) / 2 - cy
    m = cv2.warpAffine(mask, mat, (size, size),
                       flags=cv2.INTER_NEAREST).astype(np.uint8)
    signed = (cv2.distanceTransform(m, cv2.DIST_L2, 3) -
              cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)) / size
    return signed.ravel().astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=HERE.parents[1])
    p.add_argument("--manifest", type=Path, default=HERE / "folds_v2.json")
    p.add_argument("--gallery", type=Path,
                   default=HERE.parent / "cache/features/sphere42_shape.npz")
    p.add_argument("--output", type=Path, default=HERE / "cache/geometry")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    ids = manifest["gallery_model_ids"]
    bank = np.load(args.gallery)
    if list(bank["ids"]) != ids or int(bank["views"]) != 42:
        raise RuntimeError("Gallery/protocol mismatch")
    gallery = torch.from_numpy(bank["features"][:, 1].astype(np.float32)).cuda()
    gallery_sq = (gallery * gallery).mean(1)
    ranks = []
    score_rows = []
    start_time = time.monotonic()
    for i, row in enumerate(manifest["records"]):
        source = args.root / "dataset" / row["query_path"]
        query = torch.from_numpy(np.stack([
            descriptor(source, roll=angle) for angle in range(0, 360, 15)
        ]).astype(np.float32)).cuda()
        d2 = (query * query).mean(1)[:, None] + gallery_sq[None] - 2 * (query @ gallery.T) / 1024
        score = (-torch.sqrt(d2.clamp_min(0))).reshape(24, len(ids), 42).amax((0, 2)).cpu().numpy()
        score_rows.append(score)
        target = ids.index(row["model_id"])
        ranks.append(1 + int(np.count_nonzero(score > score[target])))
        if (i + 1) % 25 == 0:
            print(f"geometry {i+1}/{len(manifest['records'])}; {time.monotonic()-start_time:.1f}s", flush=True)
    scores = np.asarray(score_rows, dtype=np.float32)
    np.save(args.output / "sdf_scores.npy", scores)
    output = []
    for row, rank in zip(manifest["records"], ranks):
        output.append({"query_id": row["query_id"], "model_id": row["model_id"],
                       "group_id": row["group_id"], "outer_fold": row["outer_fold"],
                       "rank": rank})

    def metrics(values):
        a = np.asarray(values)
        return {"queries": len(a), "top1": float(np.mean(a <= 1)),
                "recall_at_5": float(np.mean(a <= 5)),
                "recall_at_20": float(np.mean(a <= 20)),
                "recall_at_50": float(np.mean(a <= 50)),
                "recall_at_100": float(np.mean(a <= 100)),
                "mrr": float(np.mean(1.0 / a)), "median_rank": float(np.median(a))}

    payload = {
        "status": "retrospective_grouped_v2_no_prospective_holdout",
        "method": "silhouette_sdf_32px_42views_24query_rolls",
        "all": metrics(ranks),
        "folds": {str(f): metrics([r["rank"] for r in output if r["outer_fold"] == f])
                  for f in range(5)},
        "queries": output,
    }
    (args.output / "sdf_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"all": payload["all"], "folds": payload["folds"]}, indent=2))


if __name__ == "__main__":
    main()
