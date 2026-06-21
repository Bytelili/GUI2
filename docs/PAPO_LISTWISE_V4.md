# PAPO grouped Listwise-v4

Listwise-v4 is a new, isolated FingerTip-20K proactive-suggestion path. It does not rename or clean v2/v3 data. The formal unit is one task group containing prompt-only messages, 2–4 candidates, an aligned target distribution, and exactly one oracle.

## Safety boundary

- Strict task JSONL files are read-only and supplied through CLI paths.
- All generated data goes to an external workspace. No command defaults to `LLaMA-Factory/data/papo`.
- Git contains code, schema, tests, templates, and synthetic fixtures only.
- Candidate generation is two-stage. Stage A emits target-free requests; Stage B validates externally generated UI-TARS/SFT results and their SHA256/provenance.
- Without imported formal candidates, only `synthetic_smoke_not_for_formal_training` releases can be built. The trainer rejects that release status.
- DPO remains out of scope until a formal v4 smoke run beats the unchanged SFT strict-holdout baseline.

## Local sequence

Run scripts 22 through 27 with explicit `--train-tasks`, `--eval-tasks`, and `--workspace` paths. Script 22 writes `manifests/source_task_manifest.json` and a separate JSONL report for every unavailable image while retaining the original path. Script 23 creates request shards or imports model results. Scripts 24/25 provide UTF-8-BOM CSV review and an immutable JSONL audit log. Script 26 builds a new timestamped release and archive. Script 27 rechecks quality, SHA256, and manifest bindings.

Script 28 is the server registration boundary. It verifies `SHA256SUMS.txt`, refuses a synthetic release unless explicitly allowed for format-only smoke checks, copies only v4 artifacts, and merges only the two new v4 dataset entries while preserving v2/v3.

The synthetic acceptance path is:

```powershell
python scripts/22_audit_proactive_tasks_v4.py --train-tasks <TRAIN_JSONL> --eval-tasks <EVAL_JSONL> --workspace <WORKSPACE>
python scripts/26_build_proactive_listwise_v4.py --train-tasks <TRAIN_JSONL> --eval-tasks <EVAL_JSONL> --workspace <WORKSPACE> --release-kind smoke_v4 --synthetic-smoke
python scripts/27_audit_proactive_listwise_v4.py --release-dir <TIMESTAMPED_RELEASE_DIR> --report-dir <WORKSPACE_REPORT_DIR> --allow-unavailable-images
```

This validates the local machinery; it is not evidence that full-v4 data exists.

## Training behavior

`use_papo_listwise` retains the old weighted sequence-NLL behavior. v4 exclusively uses `use_papo_group_listwise`. The converter and tokenizer keep each task indivisible; the collator expands its candidates together and emits local group indices. Each candidate score is mean token log-probability. Cross-entropy is computed between target and model softmax distributions separately inside every group.

The trainer requires `papo_dataset_manifest` and `papo_dataset_root`, verifies registered file hashes before training, and logs group loss, oracle top-1 accuracy, oracle margin, target entropy, and policy entropy. Packing is rejected for grouped v4 because it would obscure candidate boundaries.

## Human review

Candidate decisions are `keep`, `drop_unrelated`, `drop_popular_bias`, `drop_history_copy`, `drop_cross_user`, `regenerate`, and `manual_replace`. Duplicate annotations, unknown IDs, split/group mismatches, invalid probabilities, oracle removal, eval oracle edits, and eval replacements fail closed. Regression cases are selected first, followed by deterministic user/class/split-stratified sampling.
