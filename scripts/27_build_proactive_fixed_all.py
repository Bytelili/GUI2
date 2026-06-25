from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.config import load_config  # noqa: E402
from papo.llamafactory_export import dataset_info  # noqa: E402
from papo.proactive_fixed_export import (  # noqa: E402
    DPOExportConfig,
    RerankExportConfig,
    WeightedListwiseExportConfig,
    audit_wide_rows,
    build_examples_payload,
    export_dpo_rows,
    export_oracle_sft_rows,
    export_rerank_rows,
    export_weighted_listwise_rows,
    read_wide_csv,
    split_rows_by_user_time,
    validate_dpo_rows,
    validate_rerank_rows,
    validate_sft_rows,
    validate_weighted_listwise_rows,
    write_jsonl_dataset,
    write_report,
)
from papo.proactive_quality_gate import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all proactive-fixed exports in one pass.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--project-config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    config = load_config(args.project_config)
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = read_wide_csv(args.input)
    audit_report = audit_wide_rows(rows)
    write_report(report_dir / "train_wide_audit.json", audit_report)

    train_rows, eval_rows, split_report = split_rows_by_user_time(rows, args.eval_ratio, "user_time")

    sft_train, sft_train_report = export_oracle_sft_rows(train_rows)
    sft_eval, sft_eval_report = export_oracle_sft_rows(eval_rows)
    dpo_config = _dpo_config(config)
    dpo_train, dpo_train_report = export_dpo_rows(train_rows, dpo_config)
    dpo_eval, dpo_eval_report = export_dpo_rows(eval_rows, dpo_config)
    rerank_config = _rerank_config(config)
    rerank_train, rerank_train_report = export_rerank_rows(train_rows, rerank_config)
    rerank_eval, rerank_eval_report = export_rerank_rows(eval_rows, rerank_config)
    listwise_config = _weighted_listwise_config(config)
    listwise_train, listwise_train_report = export_weighted_listwise_rows(train_rows, listwise_config)
    listwise_eval, listwise_eval_report = export_weighted_listwise_rows(eval_rows, listwise_config)

    files = {
        "proactive_oracle_sft_train.jsonl": sft_train,
        "proactive_oracle_sft_eval.jsonl": sft_eval,
        "proactive_dpo_train.jsonl": dpo_train,
        "proactive_dpo_eval.jsonl": dpo_eval,
        "proactive_rerank_train.jsonl": rerank_train,
        "proactive_rerank_eval.jsonl": rerank_eval,
        "proactive_weighted_listwise_train.jsonl": listwise_train,
        "proactive_weighted_listwise_eval.jsonl": listwise_eval,
    }
    for name, rows_payload in files.items():
        write_jsonl_dataset(out_dir / name, rows_payload)

    validation_report = {
        "status": "passed",
        "sft_train": validate_sft_rows(sft_train),
        "sft_eval": validate_sft_rows(sft_eval),
        "dpo_train": validate_dpo_rows(dpo_train),
        "dpo_eval": validate_dpo_rows(dpo_eval),
        "rerank_train": validate_rerank_rows(rerank_train),
        "rerank_eval": validate_rerank_rows(rerank_eval),
        "listwise_train": validate_weighted_listwise_rows(listwise_train),
        "listwise_eval": validate_weighted_listwise_rows(listwise_eval),
    }
    if not all(
        section.get("passed")
        for section in validation_report.values()
        if isinstance(section, dict) and "passed" in section
    ):
        validation_report["status"] = "failed"
    write_report(report_dir / "validation_report.json", validation_report)

    write_report(
        report_dir / "pair_quality_report.json",
        {
            "split": split_report,
            "train": dpo_train_report,
            "eval": dpo_eval_report,
        },
    )
    write_report(
        report_dir / "rerank_quality_report.json",
        {
            "split": split_report,
            "train": rerank_train_report,
            "eval": rerank_eval_report,
        },
    )
    write_report(
        report_dir / "data_examples.json",
        build_examples_payload(sft_train, dpo_train, rerank_train, listwise_train),
    )

    mirrored_dir = _mirror_to_llamafactory(out_dir, config)
    report = {
        "status": validation_report["status"],
        "input": str(Path(args.input).resolve()),
        "output_dir": str(out_dir.resolve()),
        "report_dir": str(report_dir.resolve()),
        "split": split_report,
        "oracle_sft": {"train": sft_train_report, "eval": sft_eval_report},
        "dpo": {"train": dpo_train_report, "eval": dpo_eval_report},
        "rerank": {"train": rerank_train_report, "eval": rerank_eval_report},
        "weighted_listwise": {"train": listwise_train_report, "eval": listwise_eval_report},
        "llamafactory_sync_dir": str(mirrored_dir) if mirrored_dir else "",
    }
    write_report(report_dir / "build_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if validation_report["status"] != "passed":
        raise SystemExit(1)


def _dpo_config(project_config: dict[str, object]) -> DPOExportConfig:
    config = dict(project_config.get("proactive_fixed", {}).get("dpo", {}))
    return DPOExportConfig(
        max_pairs_per_row=int(config.get("max_pairs_per_row", 2)),
        min_reward_gap=float(config.get("min_reward_gap", 0.05)),
        min_char_similarity=float(config.get("min_char_similarity", 0.20)),
        max_char_similarity=float(config.get("max_char_similarity", 0.98)),
        same_user_min_similarity=float(config.get("same_user_min_similarity", 0.45)),
        same_user_max_similarity=float(config.get("same_user_max_similarity", 0.95)),
    )


def _rerank_config(project_config: dict[str, object]) -> RerankExportConfig:
    config = dict(project_config.get("proactive_fixed", {}).get("rerank", {}))
    return RerankExportConfig(
        min_candidates=int(config.get("min_candidates", 2)),
        shuffle_candidates=bool(config.get("shuffle_candidates", True)),
        seed=int(config.get("seed", 42)),
    )


def _weighted_listwise_config(project_config: dict[str, object]) -> WeightedListwiseExportConfig:
    config = dict(project_config.get("proactive_fixed", {}).get("weighted_listwise", {}))
    return WeightedListwiseExportConfig(
        temperature=float(config.get("temperature", 0.15)),
        min_context_prob=float(config.get("min_context_prob", 0.02)),
        min_oracle_prob=float(config.get("min_oracle_prob", 0.65)),
        max_oracle_prob=float(config.get("max_oracle_prob", 0.95)),
    )


def _mirror_to_llamafactory(out_dir: Path, project_config: dict[str, object]) -> Path | None:
    dataset_dir = Path(str(project_config["paths"]["llamafactory_data_dir"]))
    if not dataset_dir.exists():
        return None
    target_dir = dataset_dir / "proactive_fixed"
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*.jsonl"):
        shutil.copyfile(path, target_dir / path.name)
    dataset_info_path = dataset_dir / "dataset_info.json"
    info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else {}
    fixed_dataset_names = {
        "papo_proactive_oracle_sft_train",
        "papo_proactive_oracle_sft_eval",
        "papo_proactive_dpo_train",
        "papo_proactive_dpo_eval",
        "papo_proactive_rerank_train",
        "papo_proactive_rerank_eval",
        "papo_proactive_weighted_listwise_train",
        "papo_proactive_weighted_listwise_eval",
    }
    info.update({key: value for key, value in dataset_info().items() if key in fixed_dataset_names})
    dataset_info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    _ = read_jsonl(target_dir / "proactive_oracle_sft_train.jsonl")
    return target_dir


if __name__ == "__main__":
    main()
