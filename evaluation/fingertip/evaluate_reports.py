import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from torch.utils.tensorboard import SummaryWriter


PROACTIVE_REQUIRED = {
    "original_intent",
    "predicted_intent",
    "time",
    "token",
}

EXECUTION_REQUIRED = {
    "success",
    "origin_step",
    "real_step",
    "step_ratio",
    "up_sim",
    "down_sim",
    "similarity",
    "time",
    "token",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proactive", nargs="*", default=[])
    parser.add_argument("--execution", nargs="*", default=[])
    parser.add_argument("--output-dir", default="reports/fingertip")
    parser.add_argument(
        "--embedding-model",
        default="/home/dumike/zyy/GUI2/models/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_level(path: Path) -> str:
    match = re.search(r"level[_-]?([0-3])", path.stem.lower())
    return match.group(1) if match else "unknown"


def bootstrap_ci(values, samples, rng):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]

    if len(array) == 0:
        return None, None

    draws = rng.choice(array, size=(samples, len(array)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def numeric_summary(values, samples, rng):
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy()

    if len(array) == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    low, high = bootstrap_ci(array, samples, rng)

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "ci95_low": low,
        "ci95_high": high,
    }


def evaluate_proactive(paths, model, samples, rng, output_dir):
    frames = []

    for path_text in paths:
        path = Path(path_text)
        frame = pd.read_csv(path)
        missing = PROACTIVE_REQUIRED - set(frame.columns)

        if missing:
            print(f"SKIP proactive {path}: missing columns {sorted(missing)}")
            continue

        frame = frame.copy()
        frame["source_file"] = str(path)
        frame["level"] = infer_level(path)
        frames.append(frame)

    if not frames:
        return None, {}

    data = pd.concat(frames, ignore_index=True)
    originals = data["original_intent"].fillna("").astype(str).tolist()
    predictions = data["predicted_intent"].fillna("").astype(str).tolist()

    data["edit_similarity"] = [
        fuzz.ratio(original, prediction) / 100.0
        for original, prediction in zip(originals, predictions)
    ]

    original_embeddings = model.encode(
        originals,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    prediction_embeddings = model.encode(
        predictions,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    data["semantic_similarity"] = np.sum(
        original_embeddings * prediction_embeddings,
        axis=1,
    )
    data["official_similarity"] = (
        data["edit_similarity"] + data["semantic_similarity"]
    ) / 2.0

    data.to_csv(output_dir / "proactive_predictions_scored.csv", index=False)

    metrics = {}
    for level, group in data.groupby("level"):
        metrics[f"level_{level}"] = {
            "count": int(len(group)),
            "edit_similarity": numeric_summary(
                group["edit_similarity"], samples, rng
            ),
            "semantic_similarity": numeric_summary(
                group["semantic_similarity"], samples, rng
            ),
            "official_similarity": numeric_summary(
                group["official_similarity"], samples, rng
            ),
            "time": numeric_summary(group["time"], samples, rng),
            "token": numeric_summary(group["token"], samples, rng),
            "error_rate": float(
                group["predicted_intent"]
                .fillna("")
                .astype(str)
                .str.upper()
                .eq("ERROR")
                .mean()
            ),
        }

    return data, metrics


def evaluate_execution(paths, samples, rng, output_dir):
    frames = []

    for path_text in paths:
        path = Path(path_text)
        frame = pd.read_csv(path)
        missing = EXECUTION_REQUIRED - set(frame.columns)

        if missing:
            print(f"SKIP execution {path}: missing columns {sorted(missing)}")
            continue

        frame = frame.copy()
        frame["source_file"] = str(path)
        frames.append(frame)

    if not frames:
        return None, {}

    data = pd.concat(frames, ignore_index=True)

    for column in EXECUTION_REQUIRED:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data.to_csv(output_dir / "execution_results_scored.csv", index=False)

    metrics = {
        "count": int(len(data)),
        "success_rate": numeric_summary(data["success"], samples, rng),
        "step_ratio": numeric_summary(data["step_ratio"], samples, rng),
        "up_similarity": numeric_summary(data["up_sim"], samples, rng),
        "down_similarity": numeric_summary(data["down_sim"], samples, rng),
        "personalized_similarity": numeric_summary(
            data["similarity"], samples, rng
        ),
        "time": numeric_summary(data["time"], samples, rng),
        "token": numeric_summary(data["token"], samples, rng),
    }

    return data, metrics


def flatten_metrics(prefix, value, rows):
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}/{key}" if prefix else key
            flatten_metrics(next_prefix, child, rows)
    elif value is not None:
        rows.append({"metric": prefix, "value": value})


def write_tensorboard(metrics, output_dir):
    writer = SummaryWriter(str(output_dir / "tensorboard"))

    rows = []
    flatten_metrics("", metrics, rows)

    for row in rows:
        value = row["value"]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            writer.add_scalar(row["metric"], float(value), 0)

    writer.close()
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("Loading official semantic model:", args.embedding_model)
    model = SentenceTransformer(args.embedding_model)

    _, proactive_metrics = evaluate_proactive(
        args.proactive,
        model,
        args.bootstrap_samples,
        rng,
        output_dir,
    )

    _, execution_metrics = evaluate_execution(
        args.execution,
        args.bootstrap_samples,
        rng,
        output_dir,
    )

    report = {
        "proactive_suggestion": proactive_metrics,
        "personalized_execution": execution_metrics,
        "notes": {
            "proactive_similarity":
                "Mean of fuzzy edit similarity and semantic cosine similarity.",
            "execution_success":
                "Official script defaults success to zero; annotate or verify it before final reporting.",
            "intent_class_accuracy":
                "Not reported because it is not defined by the official implementation.",
        },
    }

    json_path = output_dir / "benchmark_metrics.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = write_tensorboard(report, output_dir)
    pd.DataFrame(rows).to_csv(
        output_dir / "benchmark_summary.csv",
        index=False,
    )

    print("\n===== Unified evaluation report =====")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nWritten:", json_path)
    print("Written:", output_dir / "benchmark_summary.csv")
    print("TensorBoard:", output_dir / "tensorboard")


if __name__ == "__main__":
    main()
