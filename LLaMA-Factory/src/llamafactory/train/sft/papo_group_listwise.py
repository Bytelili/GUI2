from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union

import torch
import torch.nn.functional as F


IGNORE_INDEX = -100


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_papo_group_dataset_binding(
    manifest_path: str,
    dataset_root: str,
    *,
    allow_nonformal_smoke: bool = False,
    allow_nonformal_retrieval: bool = False,
) -> dict:
    manifest_file, root = Path(manifest_path), Path(dataset_root)
    if not manifest_file.is_file():
        raise ValueError(f"PAPO Listwise-v4 manifest does not exist: {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read PAPO Listwise-v4 manifest: {manifest_file}: {error}") from error

    release_status = manifest.get("release_status")
    is_formal = release_status == "formal_candidate_release"
    is_engineering_smoke = (
        allow_nonformal_smoke
        and manifest.get("release_kind") == "smoke_v4"
        and release_status == "synthetic_smoke_not_for_formal_training"
        and manifest.get("formal_full_v4_complete") is False
    )
    is_retrieval_only = (
        allow_nonformal_retrieval
        and manifest.get("release_kind") == "retrieval_only_v4"
        and release_status == "retrieval_only_not_for_formal_training"
        and manifest.get("formal_full_v4_complete") is False
        and manifest.get("candidate_provenance") is None
    )
    if not is_formal and not is_engineering_smoke and not is_retrieval_only:
        raise ValueError(
            "PAPO grouped training refuses this release. Non-formal engineering smoke requires "
            "`papo_allow_nonformal_smoke: true`, while retrieval-only training requires "
            "`papo_allow_nonformal_retrieval: true` and its unchanged non-formal manifest."
        )

    hashes = manifest.get("dataset_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("PAPO Listwise-v4 manifest has no dataset hash bindings.")
    for filename, expected in hashes.items():
        dataset_file = root / filename
        if not dataset_file.is_file():
            raise ValueError(f"PAPO Listwise-v4 registered dataset is missing: {dataset_file}")
        actual = _sha256_file(dataset_file)
        if actual != expected:
            raise ValueError(
                f"PAPO Listwise-v4 dataset SHA256 mismatch: {filename}: expected={expected}, actual={actual}"
            )
    return manifest


def papo_listwise_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    r"""Legacy target-policy weighted sequence negative log likelihood."""
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2), shift_labels, ignore_index=IGNORE_INDEX, reduction="none"
    )
    valid_mask = shift_labels.ne(IGNORE_INDEX)
    sequence_losses = (token_losses * valid_mask).sum(dim=1)
    weights = weights.to(device=sequence_losses.device, dtype=sequence_losses.dtype).clamp_min(0.0)
    return (sequence_losses * weights).sum() / weights.sum().clamp_min(torch.finfo(sequence_losses.dtype).eps)


def papo_group_listwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_index: torch.Tensor,
    target_probability: torch.Tensor,
    oracle_mask: torch.Tensor,
    model_temperature: float = 1.0,
    return_metrics: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    r"""Cross entropy between target and model policies inside each candidate group."""
    if model_temperature <= 0.0:
        raise ValueError("PAPO grouped Listwise model temperature must be positive.")
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("PAPO grouped Listwise logits/labels shapes are invalid.")

    candidate_count = logits.size(0)
    if any(value.numel() != candidate_count for value in (group_index, target_probability, oracle_mask)):
        raise ValueError("PAPO grouped Listwise metadata does not match candidate count.")

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    valid_mask = shift_labels.ne(IGNORE_INDEX)
    token_count = valid_mask.sum(dim=1)
    if torch.any(token_count == 0):
        raise ValueError("PAPO grouped Listwise candidate has no valid target tokens.")

    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_log_probs = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sequence_scores = (token_log_probs * valid_mask).sum(dim=1) / token_count

    group_index = group_index.to(device=sequence_scores.device, dtype=torch.long)
    target_probability = target_probability.to(device=sequence_scores.device, dtype=sequence_scores.dtype)
    oracle_mask = oracle_mask.to(device=sequence_scores.device, dtype=torch.bool)
    if not torch.all(torch.isfinite(target_probability)) or torch.any(target_probability < 0):
        raise ValueError("PAPO target probabilities must be finite and non-negative.")

    losses, accuracies, margins, target_entropies, policy_entropies = [], [], [], [], []
    unique_groups = torch.unique(group_index, sorted=True)
    for group_id in unique_groups:
        mask = group_index.eq(group_id)
        if int(mask.sum()) < 2:
            raise ValueError("Every PAPO Listwise group must contain at least two candidates.")
        q = target_probability[mask]
        group_sum = q.sum()
        if not torch.isfinite(group_sum) or group_sum <= 0:
            raise ValueError("PAPO target probabilities must sum to a positive finite value inside every group.")
        if not torch.isclose(group_sum, q.new_tensor(1.0), atol=1e-5, rtol=1e-5):
            if torch.isclose(group_sum, q.new_tensor(1.0), atol=5e-3, rtol=5e-3):
                q = q / group_sum
            else:
                raise ValueError("PAPO target probabilities must sum to one inside every group.")
        group_oracle = oracle_mask[mask]
        if int(group_oracle.sum()) != 1:
            raise ValueError("Every PAPO Listwise group must contain exactly one oracle.")
        if q[group_oracle].item() + 1e-8 < q.max().item():
            raise ValueError("PAPO oracle must have the highest target probability.")

        scores = sequence_scores[mask] / model_temperature
        log_policy = F.log_softmax(scores, dim=0)
        policy = log_policy.exp()
        losses.append(-(q * log_policy).sum())
        oracle_position = group_oracle.nonzero(as_tuple=False).squeeze(-1)
        accuracies.append(scores.argmax().eq(oracle_position).float().squeeze())
        oracle_score = scores[group_oracle].squeeze(0)
        margins.append(oracle_score - scores[~group_oracle].max())
        target_entropies.append(-(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum())
        policy_entropies.append(-(policy * log_policy).sum())

    if not losses:
        raise ValueError("PAPO grouped Listwise batch contains no groups.")
    loss = torch.stack(losses).mean()
    metrics = {
        "group_loss": loss.detach(),
        "oracle_top1_accuracy": torch.stack(accuracies).mean().detach(),
        "oracle_margin": torch.stack(margins).mean().detach(),
        "target_entropy": torch.stack(target_entropies).mean().detach(),
        "policy_entropy": torch.stack(policy_entropies).mean().detach(),
    }
    return (loss, metrics) if return_metrics else loss


def prepare_papo_group_eval_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    r"""Force grouped PAPO eval to loss-only mode.

    Grouped PAPO validation needs only `eval_loss` plus the custom aggregated PAPO
    metrics collected inside `compute_loss`. Returning full logits/labels from the
    Hugging Face evaluation loop triggers large cross-rank gathers that are both
    unnecessary and prone to long NCCL stalls on multimodal grouped batches.
    """

    eval_kwargs = dict(kwargs)
    eval_kwargs["prediction_loss_only"] = True
    return eval_kwargs
