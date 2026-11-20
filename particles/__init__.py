"""Particle-emitter package."""

from .emitter import (
    ParticleError,
    budget_limit,
    emit,
    integrate,
    lifetime_curve,
    recycle,
    simulate_batch,
    sort_by_depth,
    spawn_deterministic,
)

__all__ = [
    "ParticleError",
    "emit",
    "integrate",
    "recycle",
    "sort_by_depth",
    "spawn_deterministic",
    "budget_limit",
    "lifetime_curve",
    "simulate_batch",
]
