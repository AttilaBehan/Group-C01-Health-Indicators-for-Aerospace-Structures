"""
Prognostic criteria (monotonicity, trendability, prognosability) and fitness.

This module is intentionally a thin re-export of ``Prog_crit`` so there is a
single source of truth. It used to be a byte-for-byte duplicate of
``Prog_crit.py``; because the hyperparameter-optimisation objective imported its
``fitness`` from here while the rest of the code imported it from ``Prog_crit``,
any drift between the two copies would have meant the optimiser minimised a
different quantity than the one reported. Keeping this as an alias makes that
impossible.
"""
from Prog_crit import (
    Pr,
    Pr_single,
    Tr,
    Mo_single,
    Mo,
    fitness,
    test_fitness,
    scale_exact,
)

__all__ = [
    "Pr",
    "Pr_single",
    "Tr",
    "Mo_single",
    "Mo",
    "fitness",
    "test_fitness",
    "scale_exact",
]
