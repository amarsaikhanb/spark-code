"""Code execution sandbox.

All generated code runs in a subprocess with timeout + memory limits, NEVER
in the same process as the trainer. The trainer is GPU-resident and untrusted
generations could OOM, infinite-loop, or import bad modules.

This package has no heavy dependencies — it uses only stdlib subprocess —
so the public API is re-exported here for convenience.
"""

from spark_code.sandbox.executor import (
    evaluate_generated_code,
    execute_code,
    execute_tests_individually,
    run_python_program,
)

__all__ = [
    "execute_code",
    "execute_tests_individually",
    "evaluate_generated_code",
    "run_python_program",
]
