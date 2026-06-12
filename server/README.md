# Server Deployment

This project is developed and smoke-tested locally, then trained on a Linux GPU
server. Do not upload local virtual environments or the separately cloned
LLaMA-Factory repository.

Expected server layout:

```text
GUI/
  LLaMA-Factory/
    data/papo/
  configs/
  data/
  scripts/
  server/
  src/
```

Setup:

```bash
bash server/setup_server.sh
export PATH="$PWD/.venv/bin:$PATH"
```

LLaMA-Factory requires Python 3.11 or newer. The setup script first looks for
`python3.12` or `python3.11`. If neither is available but Conda is installed,
it creates a Python 3.11 Conda environment at `.venv` without requiring
administrator access. To select a specific interpreter:

```bash
PYTHON_BIN=/path/to/python3.11 bash server/setup_server.sh
```

Edit the `paths` section in `config.yaml`, especially:

```yaml
paths:
  raw_root: /home/dumike/zyy/GUI/data/raw/fingertip20k
  official_root: /home/dumike/zyy/GUI/new/GUI/data/official/fingertip20k
  qwen_model_path: /home/dumike/zyy/GUI/backbone/Qwen2.5-VL-3B-Instruct
  checkpoint_root: /home/dumike/zyy/GUI/new/GUI/LLaMA-Factory/saves/papo
  logging_root: /home/dumike/zyy/GUI/new/GUI/runs/papo
```

Build all task data, PAPO targets, preference pairs, and LLaMA-Factory data:

```bash
bash server/prepare_train_data.sh
```

Validate only the configured paths:

```bash
python scripts/12_validate_config_paths.py --config config.yaml
```

Train:

```bash
bash server/train_auto_resume.sh configs/llamafactory/generated/proactive_sft.yaml
bash server/train_auto_resume.sh configs/llamafactory/generated/execution_sft.yaml
bash server/train_auto_resume.sh configs/llamafactory/generated/execution_listwise.yaml
bash server/train_auto_resume.sh configs/llamafactory/generated/execution_dpo.yaml
```

For multiple GPUs:

```bash
NUM_GPUS=8 bash server/train_auto_resume.sh configs/llamafactory/generated/execution_sft.yaml
```

For formal `*_clean_v2` configs, direct `server/train.sh` execution is blocked
until `server/train_auto_resume.sh` creates a verified preflight gate. The
launcher also refuses checkpoints whose dataset/config hashes changed and
finalizes the best evaluated checkpoint without loading it across GPUs.

The uploaded project already contains LLaMA-Factory. `prepare_train_data.sh`
creates `LLaMA-Factory/data/papo/RawDataset` as a link to the server dataset,
runs the config-driven pipeline, renders training YAML files, and validates
all exported image paths.

The bundled LLaMA-Factory contains the PAPO DPO extension. Keep this bundled
checkout when uploading: `pref_loss: papo` reads the exported `papo_weight`
column and applies normalized weighted DPO loss.

Adjust model paths, batch sizes, DeepSpeed/FSDP settings, and output paths after
the target server GPU topology is known.
