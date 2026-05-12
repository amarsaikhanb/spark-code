#!/usr/bin/env bash
# Usage:
#   bash scripts/reproduce_paper.sh A           # condition A only
#   bash scripts/reproduce_paper.sh C_regular   # best C run
#   bash scripts/reproduce_paper.sh C_light     # heavy aux ablation
#   bash scripts/reproduce_paper.sh all         # all three (sequentially)

set -euo pipefail

GPU=${GPU:-0}
RUN=${1:-all}

run_A() {
    echo "[reproduce] Condition A: exec-only GRPO"
    python scripts/run_experiment.py \
        --config configs/condition_A.json \
        --gpu "$GPU"
}

run_C_regular() {
    echo "[reproduce] Condition C-regular: GRPO + light aux (best run)"
    python scripts/run_experiment.py \
        --config configs/condition_C_regular.json \
        --gpu "$GPU"
}

run_C_light() {
    echo "[reproduce] Condition C-light: GRPO + heavy aux (ablation)"
    python scripts/run_experiment.py \
        --config configs/condition_C_light.json \
        --gpu "$GPU"
}

case "$RUN" in
    A) run_A ;;
    C_regular) run_C_regular ;;
    C_light) run_C_light ;;
    all)
        run_A
        run_C_regular
        run_C_light
        ;;
    *)
        echo "Unknown run: $RUN. Choose from: A, C_regular, C_light, all"
        exit 1
        ;;
esac

echo "[reproduce] Done."
