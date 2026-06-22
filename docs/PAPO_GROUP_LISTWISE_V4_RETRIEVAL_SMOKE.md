# PAPO Grouped Listwise-v4 Retrieval Smoke

This workflow is an engineering pre-experiment over the immutable `20260622T042745Z` smoke release. It is not
full-v4, does not use generated Draft candidates, and cannot support a formal effect claim.

## Boundaries

- The release manifest and every registered dataset file remain SHA256-bound.
- Non-formal training is rejected unless `papo_allow_nonformal_smoke: true` is set explicitly.
- The exception accepts only `smoke_v4` with `synthetic_smoke_not_for_formal_training` and
  `formal_full_v4_complete: false`.
- `use_papo_group_listwise: true`, `use_papo_listwise: false`, and `packing: false` are mandatory.
- v2/v3 data and source task JSONL files are not read for writing.
- The oracle-only control reuses the same 1000/200 prompts and only changes the objective to ordinary continuation
  SFT on the oracle answer.

## Server actions

```bash
bash server/run_papo_group_listwise_v4_retrieval_smoke.sh prepare
bash server/run_papo_group_listwise_v4_retrieval_smoke.sh train-grouped
bash server/run_papo_group_listwise_v4_retrieval_smoke.sh status
bash server/run_papo_group_listwise_v4_retrieval_smoke.sh report
bash server/run_papo_group_listwise_v4_retrieval_smoke.sh train-control
```

The grouped report contains train/eval loss, oracle top-1 accuracy, oracle margin, target entropy, policy entropy,
and scans logs for NaN/Inf, OOM, traceback, and group/candidate alignment failures.
