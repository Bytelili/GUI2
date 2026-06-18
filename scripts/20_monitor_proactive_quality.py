from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_quality_gate import normalize_text, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor proactive candidate generation and emit live quality warnings."
    )
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--output-dir", default="reports/proactive/live_quality_monitor")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-answer-frequency", type=int, default=100)
    parser.add_argument("--min-unique-candidates", type=int, default=2)
    parser.add_argument("--stop-on-block", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    candidate_file = Path(args.candidate_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_file = output_dir / "STOP_QUALITY_FAILED.txt"
    report_file = output_dir / "live_quality_report.json"

    seen_size = -1
    while True:
        if candidate_file.exists():
            size = candidate_file.stat().st_size
        else:
            size = -1

        if size != seen_size or args.once:
            seen_size = size
            report = audit_candidate_generation_file(
                candidate_file,
                max_answer_frequency=args.max_answer_frequency,
                min_unique_candidates=args.min_unique_candidates,
            )
            write_json(report_file, report)
            print_live_report(report)

            if report["status"] == "failed":
                stop_file.write_text(
                    "\n".join(report["blocking_reasons"]) + "\n",
                    encoding="utf-8",
                )
                print(f"[BLOCK] wrote stop marker: {stop_file}", flush=True)
                if args.stop_on_block:
                    raise SystemExit(1)
            elif stop_file.exists():
                stop_file.unlink()

        if args.once:
            break
        time.sleep(args.poll_seconds)


def audit_candidate_generation_file(
    path: Path,
    *,
    max_answer_frequency: int,
    min_unique_candidates: int,
) -> dict[str, Any]:
    rows = read_generation_rows(path)
    answer_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    counters = Counter()
    examples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        task_id = str(row.get("task_id") or row.get("id") or row.get("papo_episode_id") or index)
        target = first_text(row, ("target", "target_text", "original_intent", "intent", "oracle"))
        prompt = first_text(row, ("prompt", "input", "history", "user_prompt"))
        prompt_norm = normalize_text(prompt)
        candidates = extract_candidates(row)

        if not candidates:
            counters["empty_candidate_rows"] += 1
            examples.append({"issue": "empty_candidate_rows", "task_id": task_id})
            continue

        unique = {normalize_text(candidate["text"]) for candidate in candidates if normalize_text(candidate["text"])}
        if len(unique) < min_unique_candidates:
            counters["low_unique_candidate_rows"] += 1
            examples.append(
                {
                    "issue": "low_unique_candidate_rows",
                    "task_id": task_id,
                    "unique": len(unique),
                    "target": target,
                }
            )

        has_oracle = False
        for candidate in candidates:
            text = candidate["text"]
            text_norm = normalize_text(text)
            source = candidate["source"]
            source_counts[source] += 1
            if text_norm:
                answer_counts[text_norm] += 1
            if target and text_norm == normalize_text(target):
                has_oracle = True
            if prompt_norm and text_norm and len(text_norm) >= 4 and text_norm in prompt_norm:
                counters["candidate_prompt_copy_hits"] += 1
                if len(examples) < 100:
                    examples.append(
                        {
                            "issue": "candidate_prompt_copy_hits",
                            "task_id": task_id,
                            "source": source,
                            "candidate": text,
                        }
                    )

        if target and not has_oracle:
            counters["missing_oracle_rows"] += 1
            if len(examples) < 100:
                examples.append({"issue": "missing_oracle_rows", "task_id": task_id, "target": target})

    repeated = [(answer, count) for answer, count in answer_counts.most_common(30) if count > max_answer_frequency]
    blocking_reasons: list[str] = []
    warning_reasons: list[str] = []

    for key in ("empty_candidate_rows", "missing_oracle_rows"):
        if counters[key] > 0:
            blocking_reasons.append(f"{key}={counters[key]}")
    if counters["candidate_prompt_copy_hits"] > 0:
        warning_reasons.append(f"candidate_prompt_copy_hits={counters['candidate_prompt_copy_hits']}")
    if repeated:
        warning_reasons.append(f"popular_answers_over_cap={len(repeated)}")

    return {
        "status": "failed" if blocking_reasons else "warning" if warning_reasons else "passed",
        "path": str(path),
        "exists": path.exists(),
        "rows": len(rows),
        "counters": dict(counters),
        "source_counts": dict(source_counts),
        "top_repeated_answers": [{"answer": answer, "count": count} for answer, count in repeated],
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "examples": examples[:100],
    }


def read_generation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows
    data = json.loads(text)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "records", "items", "examples"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
        return [data]
    return []


def extract_candidates(row: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    raw = row.get("candidates") or row.get("options") or row.get("model_candidates") or []
    if not isinstance(raw, list):
        return candidates
    for item in raw:
        if isinstance(item, str):
            candidates.append({"text": item, "source": "unknown"})
        elif isinstance(item, dict):
            text = first_text(item, ("text", "candidate", "prediction", "answer", "intent", "response", "output"))
            source = str(item.get("source") or item.get("candidate_source") or item.get("source_type") or "unknown")
            candidates.append({"text": text, "source": source})
    return candidates


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        text = flatten_text(value)
        if text:
            return text
    return ""


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := flatten_text(item)))
    if isinstance(value, dict):
        for key in ("content", "value", "text", "intent", "answer", "candidate", "prediction"):
            text = flatten_text(value.get(key))
            if text:
                return text
    return ""


def print_live_report(report: dict[str, Any]) -> None:
    status = str(report["status"]).upper()
    print(
        f"[LIVE QUALITY] status={status} rows={report['rows']} "
        f"counters={report['counters']} warnings={report['warning_reasons']} blocks={report['blocking_reasons']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
