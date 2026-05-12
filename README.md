# spark-code

Code and experiments for the *Self-Improving AI Code Generation Through Co-Evolving Policy and Reward*. This repository adapts the [SPARK framework](https://arxiv.org/abs/2509.22624) (Liu et al., 2025) from mathematical reasoning to code generation, where execution provides both deterministic verification and rich diagnostic feedback.

## Quick start

```bash
git clone https://github.com/amarsaikhanb/spark-code.git
cd spark-code
pip install -e .

# Smoke test (no training)
python scripts/run_experiment.py --gpu 0 --condition C --output ./smoke \
    --iterations 1 --rollouts 4 --train-problems 20 --eval-samples 2 \
    --skip-frontier --no-wandb --reflection-eval-problems 10
```

To reproduce the headline runs:

```bash
# Condition A: exec-only GRPO baseline
python scripts/run_experiment.py --config configs/condition_A.json --gpu 0

# Condition C-regular: SPARK-style co-evolution (best run)
python scripts/run_experiment.py --config configs/condition_C_regular.json --gpu 1
```

## What's in this repo

```
spark_code/        # importable Python package (data, sandbox, model, training, eval, utils)
scripts/           # CLI entrypoint and bash wrappers
configs/           # JSON configs for the three published runs
results/           # coming soon
tests/             # pytest suite (sandbox, extraction, metrics)
```

## Citation

If you use this code or build on this work:

```bibtex
@misc{batjargal2026sparkcode,
  author = {Batjargal, Amarsaikhan},
  title  = {Self-Improving AI Code Generation Through Co-Evolving Policy and Reward},
  year   = {2026},
  url    = {https://github.com/amarsaikhanb/spark-code}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
