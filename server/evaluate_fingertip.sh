#!/usr/bin/env bash
set -uo pipefail

cd /home/dumike/zyy/GUI2
source server_env.sh

mapfile -t PROACTIVE_RESULTS < <(
    find reports results result_suggestion evaluation \
      -type f -iname '*.csv' 2>/dev/null |
    while read -r file; do
        if head -n 1 "$file" | grep -q 'original_intent.*predicted_intent'; then
            printf '%s\n' "$file"
        fi
    done |
    sort -u
)

mapfile -t EXECUTION_RESULTS < <(
    find reports results result_execution evaluation \
      -type f -iname '*.csv' 2>/dev/null |
    while read -r file; do
        if head -n 1 "$file" | grep -q 'success.*origin_step.*real_step'; then
            printf '%s\n' "$file"
        fi
    done |
    sort -u
)

echo "Proactive result files: ${#PROACTIVE_RESULTS[@]}"
printf '  %s\n' "${PROACTIVE_RESULTS[@]}"

echo "Execution result files: ${#EXECUTION_RESULTS[@]}"
printf '  %s\n' "${EXECUTION_RESULTS[@]}"

python evaluation/fingertip/evaluate_reports.py \
    --proactive "${PROACTIVE_RESULTS[@]}" \
    --execution "${EXECUTION_RESULTS[@]}" \
    --output-dir reports/fingertip

STATUS=$?

echo "Evaluator exit status: $STATUS"
echo "Generated report files:"
find reports/fingertip -maxdepth 3 -type f -printf '%p  %k KB\n' 2>/dev/null |
sort

echo "Terminal remains open."
