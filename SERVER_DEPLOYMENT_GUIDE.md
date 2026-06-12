# Server Deployment Guide

## Expected Server Paths

```text
/home/dumike/zyy/GUI/new/GUI                         project
/home/dumike/zyy/GUI/data/raw/fingertip20k          FingerTip raw data
/home/dumike/zyy/GUI/backbone/Qwen2.5-VL-3B-Instruct model
```

The code archive intentionally excludes the raw dataset, model weights,
generated predictions, visualization outputs, checkpoints, and virtual
environments.

## Upload From Windows PowerShell

Replace `SERVER_HOST` with the server hostname or IP:

```powershell
$server="dumike@SERVER_HOST"
scp "C:\Users\lenovo\Desktop\GUI\dist\GUI_server_bundle.zip" "${server}:/home/dumike/zyy/GUI/new/"
```

Upload the large raw dataset separately with resumable `rsync` from WSL. The
trailing slashes are important because they copy the contents of `Raw` into
the configured `fingertip20k` directory:

```bash
ssh dumike@SERVER_HOST "mkdir -p /home/dumike/zyy/GUI/data/raw/fingertip20k"
rsync -avhP --partial /mnt/d/0608DataSet/Raw/ dumike@SERVER_HOST:/home/dumike/zyy/GUI/data/raw/fingertip20k/
```

## Unpack, Configure, and Validate

Run on the Linux server:

```bash
set -euo pipefail
mkdir -p /home/dumike/zyy/GUI/new /home/dumike/zyy/GUI/data/raw/fingertip20k
cd /home/dumike/zyy/GUI/new
unzip -q -o GUI_server_bundle.zip
cd GUI

export PAPO_PROJECT_ROOT=/home/dumike/zyy/GUI/new/GUI
export PAPO_RAW_ROOT=/home/dumike/zyy/GUI/data/raw/fingertip20k
export FINGERTIP_ROOT=/home/dumike/zyy/GUI/data/raw/fingertip20k
export FINGERTIP_ARCHIVE_ROOT=/home/dumike/zyy/GUI/data/raw/fingertip20k
export FINGERTIP_OFFICIAL_ROOT=/home/dumike/zyy/GUI/new/GUI/data/official/fingertip20k
export QWEN_MODEL_PATH=/home/dumike/zyy/GUI/backbone/Qwen2.5-VL-3B-Instruct
export PAPO_WORK_DIR=/home/dumike/zyy/GUI/new/GUI/data/papo_config_run
export PAPO_TASK_DIR=/home/dumike/zyy/GUI/new/GUI/data/papo_tasks
export PAPO_CHECKPOINT_ROOT=/home/dumike/zyy/GUI/new/GUI/LLaMA-Factory/saves/papo
export PAPO_LOGGING_ROOT=/home/dumike/zyy/GUI/new/GUI/runs/papo

bash server/setup_server.sh
export PATH="$PWD/.venv/bin:$PATH"
python scripts/12_validate_config_paths.py --config config.yaml --create_output_dirs
python scripts/13_smoke_test_papo_objective.py
```

## Prepare Full Training Data

```bash
cd /home/dumike/zyy/GUI/new/GUI
export PATH="$PWD/.venv/bin:$PATH"
export PAPO_RAW_ROOT=/home/dumike/zyy/GUI/data/raw/fingertip20k
export QWEN_MODEL_PATH=/home/dumike/zyy/GUI/backbone/Qwen2.5-VL-3B-Instruct

RAW_ROOT="$PAPO_RAW_ROOT" bash server/prepare_train_data.sh 2>&1 | tee prepare_train_data.log
```

## Train

Run the stages in order. Each later stage loads the adapter produced by the
previous stage.

```bash
cd /home/dumike/zyy/GUI/new/GUI
export PATH="$PWD/.venv/bin:$PATH"

bash server/train.sh configs/llamafactory/generated/execution_sft.yaml 2>&1 | tee train_execution_sft.log
bash server/train.sh configs/llamafactory/generated/execution_listwise.yaml 2>&1 | tee train_execution_listwise.log
bash server/train.sh configs/llamafactory/generated/execution_dpo.yaml 2>&1 | tee train_execution_dpo.log
```

For multi-GPU training, prefix a training command with `NUM_GPUS=8`.
