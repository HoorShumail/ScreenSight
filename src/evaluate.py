"""
Zero-shot evaluation of ScreenSightModel on the ScreenSpot benchmark.

Usage:
    python -m src.evaluate                  # full test set (~1,200 samples)
    python -m src.evaluate --limit 100       # quick smoke test
    python -m src.evaluate --out my_run.json
"""

import argparse
import json
import time
from collections import defaultdict

from datasets import load_dataset

from .model import ScreenSightModel

DATASET_ID = "rootsautomation/ScreenSpot"


def point_in_bbox(x: float, y: float, bbox) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def run_eval(limit: int = None, out_path: str = "results.json"):
    print(f"Loading {DATASET_ID} ...")
    dataset = load_dataset(DATASET_ID)["test"]
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    print("Loading Qwen2.5-VL-3B-Instruct (first run downloads ~6GB) ...")
    model = ScreenSightModel()

    per_sample = []
    correct_by_group = defaultdict(lambda: [0, 0])  # group -> [correct, total]

    start = time.time()
    n = len(dataset)
    for i, sample in enumerate(dataset):
        image = sample["image"]
        instruction = sample["instruction"]
        bbox = sample["bbox"]
        data_type = sample["data_type"]
        data_source = sample["data_source"]

        pred = model.predict_click(image, instruction)
        hit = pred is not None and point_in_bbox(pred[0], pred[1], bbox)

        per_sample.append(
            {
                "index": i,
                "instruction": instruction,
                "bbox": bbox,
                "prediction": pred,
                "correct": hit,
                "data_type": data_type,
                "data_source": data_source,
            }
        )

        for group_key in (f"type:{data_type}", f"source:{data_source}", "overall"):
            correct_by_group[group_key][1] += 1
            if hit:
                correct_by_group[group_key][0] += 1

        if (i + 1) % 20 == 0 or (i + 1) == n:
            elapsed = time.time() - start
            print(f"  {i + 1}/{n}  ({elapsed:.0f}s elapsed)")

    summary = {
        group: {"correct": c, "total": t, "accuracy": round(c / t, 4) if t else 0.0}
        for group, (c, t) in correct_by_group.items()
    }

    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2)

    print("\n=== Results ===")
    for group, stats in sorted(summary.items()):
        print(f"{group:20s}  {stats['accuracy']*100:5.1f}%  ({stats['correct']}/{stats['total']})")
    print(f"\nSaved full results (including every prediction) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on a subset for a quick smoke test")
    parser.add_argument("--out", type=str, default="results.json")
    args = parser.parse_args()
    run_eval(limit=args.limit, out_path=args.out)
