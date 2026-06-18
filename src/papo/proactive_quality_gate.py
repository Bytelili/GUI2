from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BLOCK_THRESHOLDS = {
    "missing_oracle_groups": 0,
    "oracle_not_top1_groups": 0,
    "prompt_leak_high_weight_rows": 0,
    "missing_image_rows": 0,
    "empty_candidate_groups": 0,
    "weak_margin_groups": 0,
}

DEFAULT_WARN_THRESHOLDS = {
    "repeated_answer_max_count": 100,
    "history_source_top1_groups": 0,
    "mean_non_oracle_mass": 0.35,
}


@dataclass
class QualityIssue:
    severity: str
    category: str
    item_id: str
    detail: str
    evidence: str = ""


@dataclass
class QualityDecision:
    status: str
    blocking_reasons: list[str]
    warning_reasons: list[str]


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s+", "", text)
    return text.strip().lower()


def read_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list: {path}")
    return [row for row in data if isinstance(row, dict)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_issues_csv(path: Path, issues: list[QualityIssue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["severity", "category", "item_id", "detail", "evidence"])
        writer.writeheader()
        for issue in issues:
            writer.writerow(asdict(issue))


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := extract_text(item)))
    if isinstance(value, dict):
        for key in (
            "content",
            "value",
            "text",
            "answer",
            "candidate",
            "candidate_text",
            "intent",
            "target",
            "chosen",
            "rejected",
            "prediction",
            "response",
        ):
            if key in value:
                text = extract_text(value[key])
                if text:
                    return text
        if "messages" in value:
            return extract_text(value["messages"])
        if "conversations" in value:
            return extract_text(value["conversations"])
    return ""


def prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or row.get("conversations") or []
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        if role not in {"assistant", "gpt", "model"}:
            parts.append(extract_text(message))
    return "\n".join(parts)


def assistant_text(row: dict[str, Any]) -> str:
    for key in ("answer", "candidate", "candidate_text", "response", "output"):
        if key in row:
            text = extract_text(row[key])
            if text:
                return text
    messages = row.get("messages") or row.get("conversations") or []
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or message.get("from") or "").lower()
            if role in {"assistant", "gpt", "model"}:
                return extract_text(message)
    return ""


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def row_group_id(row: dict[str, Any], index: int) -> str:
    metadata = row_metadata(row)
    for key in (
        "group_id",
        "preference_group_id",
        "task_id",
        "papo_episode_id",
        "episode_id",
        "sample_id",
    ):
        value = metadata.get(key) or row.get(key)
        if value:
            return str(value)
    return f"__row_{index}"


def row_source(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    for key in ("candidate_source", "source", "source_type", "origin"):
        value = metadata.get(key) or row.get(key)
        if value:
            return str(value)
    return "unknown"


def row_weight(row: dict[str, Any]) -> float:
    for key in ("papo_listwise_weight", "listwise_weight", "weight", "score", "reward"):
        value = row.get(key)
        if value is None:
            value = row_metadata(row).get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def is_oracle_row(row: dict[str, Any], answer: str, target: str) -> bool:
    source = row_source(row).lower()
    if source in {"oracle", "oracle_target", "target", "gold", "ground_truth"}:
        return True
    return bool(target and normalize_text(answer) == normalize_text(target))


def row_target(row: dict[str, Any]) -> str:
    metadata = row_metadata(row)
    for source in (metadata, row):
        for key in ("target", "target_text", "oracle", "oracle_target", "original_intent", "intent"):
            value = source.get(key)
            if value:
                return extract_text(value)
    return ""


def image_refs(row: dict[str, Any]) -> list[str]:
    images: list[str] = []
    for key in ("images", "image", "image_paths", "image_path"):
        value = row.get(key)
        if isinstance(value, str):
            images.append(value)
        elif isinstance(value, list):
            images.extend(str(item) for item in value if isinstance(item, str))
    return images


def resolve_image(path_text: str, roots: list[Path]) -> bool:
    path = Path(path_text.replace("file://", ""))
    if path.is_absolute():
        return path.exists()
    return any((root / path).exists() for root in roots)


class ProactiveQualityGate:
    def __init__(
        self,
        *,
        image_roots: list[Path] | None = None,
        min_oracle_margin: float = 0.10,
        max_answer_frequency: int = 100,
        max_non_oracle_mass: float = 0.35,
        leak_weight_threshold: float = 0.05,
        progress_every: int = 1000,
        fail_fast: bool = False,
    ) -> None:
        self.image_roots = image_roots or []
        self.min_oracle_margin = min_oracle_margin
        self.max_answer_frequency = max_answer_frequency
        self.max_non_oracle_mass = max_non_oracle_mass
        self.leak_weight_threshold = leak_weight_threshold
        self.progress_every = progress_every
        self.fail_fast = fail_fast
        self.issues: list[QualityIssue] = []

    def add_issue(self, severity: str, category: str, item_id: str, detail: str, evidence: Any = "") -> None:
        issue = QualityIssue(severity, category, item_id, detail, str(evidence)[:400])
        self.issues.append(issue)
        prefix = "BLOCK" if severity == "block" else "WARN"
        print(f"[{prefix}] {category} | {item_id} | {detail} | {issue.evidence}", flush=True)
        if self.fail_fast and severity == "block":
            raise RuntimeError(f"quality gate blocked: {category} {item_id} {detail}")

    def audit_listwise(self, rows: list[dict[str, Any]], name: str = "listwise") -> dict[str, Any]:
        groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row_group_id(row, index)].append((index, row))

        answer_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        source_top1_counts: Counter[str] = Counter()
        margins: list[float] = []
        oracle_weights: list[float] = []
        non_oracle_masses: list[float] = []

        counters = Counter()
        for group_index, (group_id, group_items) in enumerate(groups.items(), start=1):
            if self.progress_every and group_index % self.progress_every == 0:
                print(
                    f"[QUALITY] {name}: scanned groups={group_index}/{len(groups)} "
                    f"blocks={sum(i.severity == 'block' for i in self.issues)} "
                    f"warnings={sum(i.severity == 'warn' for i in self.issues)}",
                    flush=True,
                )

            if not group_items:
                counters["empty_candidate_groups"] += 1
                self.add_issue("block", "empty_candidate_group", group_id, "No candidates in listwise group")
                continue

            prompt = prompt_text(group_items[0][1])
            prompt_norm = normalize_text(prompt)
            target = row_target(group_items[0][1])
            scored: list[dict[str, Any]] = []

            for row_index, row in group_items:
                answer = assistant_text(row)
                answer_norm = normalize_text(answer)
                weight = row_weight(row)
                source = row_source(row)
                source_counts[source] += 1
                if answer_norm:
                    answer_counts[answer_norm] += 1

                if image_refs(row) and self.image_roots:
                    missing = [img for img in image_refs(row) if not resolve_image(img, self.image_roots)]
                    if missing:
                        counters["missing_image_rows"] += 1
                        self.add_issue("block", "missing_image", f"{group_id}:{row_index}", "Image path cannot be resolved", missing[0])

                leak = bool(answer_norm and len(answer_norm) >= 4 and answer_norm in prompt_norm)
                if leak and weight >= self.leak_weight_threshold and source.lower() not in {"oracle", "oracle_target"}:
                    counters["prompt_leak_high_weight_rows"] += 1
                    self.add_issue(
                        "block",
                        "prompt_leak_high_weight_candidate",
                        f"{group_id}:{row_index}",
                        f"candidate appears in prompt/history with weight={weight:.6f}, source={source}",
                        answer,
                    )

                scored.append(
                    {
                        "row_index": row_index,
                        "row": row,
                        "answer": answer,
                        "weight": weight,
                        "source": source,
                        "is_oracle": is_oracle_row(row, answer, target),
                        "leak": leak,
                    }
                )

            oracle_items = [item for item in scored if item["is_oracle"]]
            if not oracle_items:
                counters["missing_oracle_groups"] += 1
                self.add_issue("block", "missing_oracle", group_id, "No oracle candidate in group", target)
                continue

            oracle = max(oracle_items, key=lambda item: float(item["weight"]))
            oracle_weight = float(oracle["weight"])
            oracle_weights.append(oracle_weight)
            top = max(scored, key=lambda item: float(item["weight"]))
            source_top1_counts[str(top["source"])] += 1
            if str(top["source"]).lower() in {"same_user_history", "history", "user_history"}:
                counters["history_source_top1_groups"] += 1
                self.add_issue("warn", "history_source_top1", group_id, "History candidate is top1", top["answer"])
            if not bool(top["is_oracle"]):
                counters["oracle_not_top1_groups"] += 1
                self.add_issue(
                    "block",
                    "oracle_not_top1",
                    group_id,
                    f"oracle_weight={oracle_weight:.6f}, top_weight={float(top['weight']):.6f}, top_source={top['source']}",
                    top["answer"],
                )

            neg_weights = [float(item["weight"]) for item in scored if not bool(item["is_oracle"])]
            if neg_weights:
                best_negative = max(neg_weights)
                margin = oracle_weight - best_negative
                margins.append(margin)
                if margin < self.min_oracle_margin:
                    counters["weak_margin_groups"] += 1
                    self.add_issue(
                        "block",
                        "weak_oracle_margin",
                        group_id,
                        f"margin={margin:.6f} < {self.min_oracle_margin:.6f}",
                        f"oracle_weight={oracle_weight:.6f}, best_negative={best_negative:.6f}",
                    )
            positive_mass = sum(float(item["weight"]) for item in scored if float(item["weight"]) > 0)
            if positive_mass > 0:
                non_oracle_mass = sum(
                    float(item["weight"])
                    for item in scored
                    if not bool(item["is_oracle"]) and float(item["weight"]) > 0
                ) / positive_mass
                non_oracle_masses.append(non_oracle_mass)
                if non_oracle_mass > self.max_non_oracle_mass:
                    counters["large_non_oracle_mass_groups"] += 1
                    self.add_issue(
                        "warn",
                        "large_non_oracle_mass",
                        group_id,
                        f"non_oracle_mass={non_oracle_mass:.6f} > {self.max_non_oracle_mass:.6f}",
                    )

        repeated = [(answer, count) for answer, count in answer_counts.most_common(50) if count > self.max_answer_frequency]
        for answer, count in repeated[:30]:
            counters["popular_answer_exceeds_cap"] += 1
            self.add_issue("warn", "popular_answer_exceeds_cap", name, f"answer frequency={count} > {self.max_answer_frequency}", answer)

        return {
            "name": name,
            "rows": len(rows),
            "groups": len(groups),
            "counters": dict(counters),
            "source_counts": dict(source_counts),
            "source_top1_counts": dict(source_top1_counts),
            "oracle_weight": summarize_numbers(oracle_weights),
            "oracle_margin": summarize_numbers(margins),
            "non_oracle_mass": summarize_numbers(non_oracle_masses),
            "top_repeated_answers": [{"answer": answer, "count": count} for answer, count in repeated[:30]],
        }

    def audit_dpo(self, rows: list[dict[str, Any]], name: str = "dpo") -> dict[str, Any]:
        counters = Counter()
        answer_counts: Counter[str] = Counter()
        margins: list[float] = []
        chosen_sources: Counter[str] = Counter()
        rejected_sources: Counter[str] = Counter()

        for index, row in enumerate(rows):
            if self.progress_every and (index + 1) % self.progress_every == 0:
                print(
                    f"[QUALITY] {name}: scanned rows={index + 1}/{len(rows)} "
                    f"blocks={sum(i.severity == 'block' for i in self.issues)} "
                    f"warnings={sum(i.severity == 'warn' for i in self.issues)}",
                    flush=True,
                )
            metadata = row_metadata(row)
            item_id = str(metadata.get("task_id") or metadata.get("papo_episode_id") or row.get("task_id") or index)
            prompt = prompt_text(row)
            prompt_norm = normalize_text(prompt)
            chosen = extract_text(row.get("chosen"))
            rejected = extract_text(row.get("rejected"))
            chosen_norm = normalize_text(chosen)
            rejected_norm = normalize_text(rejected)
            chosen_source = str(metadata.get("chosen_source") or metadata.get("candidate_source") or "unknown")
            rejected_source = str(metadata.get("rejected_source") or "unknown")
            chosen_sources[chosen_source] += 1
            rejected_sources[rejected_source] += 1
            if chosen_norm:
                answer_counts[chosen_norm] += 1
            if not chosen_norm or not rejected_norm:
                counters["empty_dpo_side_rows"] += 1
                self.add_issue("block", "empty_dpo_side", item_id, "chosen or rejected is empty")
            if chosen_norm and chosen_norm == rejected_norm:
                counters["same_chosen_rejected_rows"] += 1
                self.add_issue("block", "same_chosen_rejected", item_id, "chosen equals rejected", chosen)
            if rejected_norm and rejected_norm in prompt_norm:
                counters["rejected_prompt_leak_rows"] += 1
                self.add_issue("warn", "rejected_prompt_leak", item_id, "rejected text appears in prompt/history", rejected)
            if chosen_norm and chosen_norm in prompt_norm and chosen_source.lower() not in {"oracle", "oracle_target"}:
                counters["chosen_prompt_leak_rows"] += 1
                self.add_issue("block", "chosen_prompt_leak", item_id, "chosen non-oracle text appears in prompt/history", chosen)

            margin = None
            for key in ("preference_margin", "reward_gap", "advantage_gap"):
                value = metadata.get(key) if key in metadata else row.get(key)
                if value is not None:
                    try:
                        margin = float(value)
                        break
                    except (TypeError, ValueError):
                        pass
            if margin is not None and math.isfinite(margin):
                margins.append(margin)
                if margin < self.min_oracle_margin:
                    counters["weak_margin_pairs"] += 1
                    self.add_issue("block", "weak_dpo_margin", item_id, f"margin={margin:.6f} < {self.min_oracle_margin:.6f}")

        repeated = [(answer, count) for answer, count in answer_counts.most_common(50) if count > self.max_answer_frequency]
        for answer, count in repeated[:30]:
            counters["popular_chosen_exceeds_cap"] += 1
            self.add_issue("warn", "popular_chosen_exceeds_cap", name, f"chosen frequency={count} > {self.max_answer_frequency}", answer)

        return {
            "name": name,
            "rows": len(rows),
            "counters": dict(counters),
            "chosen_source_counts": dict(chosen_sources),
            "rejected_source_counts": dict(rejected_sources),
            "margin": summarize_numbers(margins),
            "top_repeated_chosen": [{"answer": answer, "count": count} for answer, count in repeated[:30]],
        }

    def decide(self, summaries: list[dict[str, Any]]) -> QualityDecision:
        total = Counter()
        for summary in summaries:
            total.update(summary.get("counters", {}))

        blocking: list[str] = []
        warnings: list[str] = []
        for key, allowed in DEFAULT_BLOCK_THRESHOLDS.items():
            value = int(total.get(key, 0))
            if value > allowed:
                blocking.append(f"{key}={value} > {allowed}")

        for summary in summaries:
            non_oracle_mass = summary.get("non_oracle_mass", {}).get("mean")
            if isinstance(non_oracle_mass, (int, float)) and non_oracle_mass > self.max_non_oracle_mass:
                warnings.append(f"{summary.get('name')}: mean_non_oracle_mass={non_oracle_mass:.6f}")
            repeated = summary.get("top_repeated_answers") or summary.get("top_repeated_chosen") or []
            for item in repeated[:5]:
                warnings.append(f"{summary.get('name')}: repeated_answer_count={item.get('count')}")

        for key, allowed in DEFAULT_WARN_THRESHOLDS.items():
            value = total.get(key)
            if key == "mean_non_oracle_mass":
                continue
            if isinstance(value, (int, float)) and value > allowed:
                warnings.append(f"{key}={value} > {allowed}")

        status = "failed" if blocking else "warning" if warnings else "passed"
        return QualityDecision(status=status, blocking_reasons=blocking, warning_reasons=warnings)


def summarize_numbers(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    clean.sort()
    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean),
        "min": clean[0],
        "p05": clean[int((len(clean) - 1) * 0.05)],
        "p50": clean[int((len(clean) - 1) * 0.50)],
        "p95": clean[int((len(clean) - 1) * 0.95)],
        "max": clean[-1],
    }


def print_decision(decision: QualityDecision, file: Any = sys.stdout) -> None:
    print(f"QUALITY STATUS: {decision.status.upper()}", file=file)
    if decision.blocking_reasons:
        print("Blocking reasons:", file=file)
        for reason in decision.blocking_reasons:
            print(f"  - {reason}", file=file)
    if decision.warning_reasons:
        print("Warning reasons:", file=file)
        for reason in decision.warning_reasons[:50]:
            print(f"  - {reason}", file=file)
