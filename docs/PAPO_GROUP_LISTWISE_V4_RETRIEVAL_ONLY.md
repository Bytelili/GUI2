# PAPO Grouped Listwise-v4 Retrieval-Only Experiment

This experiment uses the full adjusted retrieval candidate pools, not the 1000/200 smoke sample. It contains no
model-generated candidates and is therefore explicitly not full-v4.

## Immutable release

- Release ID: `20260622T065645Z`
- Source tasks: train 14,829, eval 824
- Eligible grouped tasks: train 13,020, eval 824
- Excluded train tasks without a retained same-user candidate: 1,809
- Final candidates: 38,899
- Group sizes: 2-3
- Cross-user candidates entering grouped loss: 0
- Quality blocks: 0
- Local-only warning: server image paths are unavailable on Windows

The manifest binds source task hashes, retrieval pool hashes, grouped dataset hashes, the quality report, and the
selection report. Training requires `papo_allow_nonformal_retrieval: true`. The smoke gate does not authorize this
release.

## Server workflow

```bash
bash server/run_papo_group_listwise_v4_retrieval_only.sh prepare
bash server/run_papo_group_listwise_v4_retrieval_only.sh train
bash server/run_papo_group_listwise_v4_retrieval_only.sh status
bash server/run_papo_group_listwise_v4_retrieval_only.sh report
```

The configuration starts from the UI-TARS-7B proactive SFT adapter, evaluates before training, runs one epoch, and
records grouped loss, oracle top-1 accuracy, oracle margin, target entropy, and policy entropy.
