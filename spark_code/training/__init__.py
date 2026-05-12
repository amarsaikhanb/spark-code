"""Training: rollouts, frontier filtering, GRPO, auxiliary SFT, condition runner.

Every module in this package depends on torch and transformers, so this
``__init__``: submodules directly:

    from spark_code.training.runner import run_condition
    from spark_code.training.rollouts import generate_rollouts
    from spark_code.training.grpo import compute_advantages, grpo_step
    from spark_code.training.auxiliary import build_aux_data, sft_step
    from spark_code.training.frontier import frontier_filter
    from spark_code.training.logprobs import get_token_logprobs
"""
