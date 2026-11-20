"""Red tests for the particle-emitter bugfix task.

All positions are world-space, all time values are seconds, and floating-point
assertions use an absolute tolerance of ``1e-6`` unless an exact endpoint is
part of the contract.  No test sleeps, uses the network, or relies on global
random state.
"""

from __future__ import annotations

import copy
import math

import pytest

from particles.emitter import (
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


def _config(**overrides) -> dict:
    config = {
        "rate": 60.0,
        "life": 2.0,
        "capacity": 128,
        "seed": 20261119,
    }
    config.update(overrides)
    return config


def _particle(
    particle_id: int,
    pos: list[float],
    vel: list[float] | None = None,
    age: float = 0.0,
    life: float = 10.0,
    radius: float = 1.0,
) -> dict:
    return {
        "id": particle_id,
        "pos": list(pos),
        "vel": [0.0, 0.0, 0.0] if vel is None else list(vel),
        "age": age,
        "life": life,
        "radius": radius,
    }


def test_emit_low_rate_accumulates_instead_of_truncating():
    config = _config(rate=0.4, life=4.0, capacity=16, seed=7)
    state = {"accumulator": 0.0, "next_id": 0}

    counts = [len(emit(state, 1.0, config)) for _ in range(5)]

    assert sum(counts) == 2
    assert state["accumulator"] == pytest.approx(0.0, abs=1e-9)
    assert state["next_id"] == 2


def test_emit_keeps_fractional_remainder_at_60fps():
    config = _config(rate=90.0, life=2.0, capacity=512, seed=8)
    state = {"accumulator": 0.0, "next_id": 0}
    frame_dt = 1.0 / 60.0

    total = sum(len(emit(state, frame_dt, config)) for _ in range(60))

    assert total == 90
    assert state["accumulator"] == pytest.approx(0.0, abs=1e-9)


def test_integrate_uses_semi_implicit_euler():
    particle = _particle(1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], life=10.0)

    updated = integrate([particle], 1.0, [0.0, -10.0, 0.0], 0.0)

    assert updated[0]["vel"] == pytest.approx([0.0, -10.0, 0.0], abs=1e-6)
    assert updated[0]["pos"] == pytest.approx([0.0, -10.0, 0.0], abs=1e-6)
    assert particle["pos"] == [0.0, 0.0, 0.0]


def test_integrate_drag_is_frame_rate_independent():
    particle = _particle(1, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])

    one_step = integrate([particle], 1.0, [0.0, 0.0, 0.0], 2.0)
    many_steps = [_particle(2, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0])]
    for _ in range(10):
        many_steps = integrate(many_steps, 0.1, [0.0, 0.0, 0.0], 2.0)

    expected = 10.0 * math.exp(-2.0)
    assert one_step[0]["vel"][0] == pytest.approx(expected, abs=1e-9)
    assert many_steps[0]["vel"][0] == pytest.approx(expected, abs=1e-9)
    assert one_step[0]["vel"][0] < 10.0


def test_recycle_keeps_newest_particles_when_full():
    particles = [_particle(index, [float(index), 0.0, 0.0]) for index in range(1, 6)]

    kept = recycle(particles, 3)

    assert [particle["id"] for particle in kept] == [3, 4, 5]
    assert [particle["id"] for particle in particles] == [1, 2, 3, 4, 5]


def test_sort_by_depth_is_far_to_near_and_stable():
    camera = {"pos": [0.0, 0.0, 10.0]}
    far = _particle(1, [0.0, 0.0, 13.0])
    near = _particle(2, [0.0, 0.0, 11.0])
    middle = _particle(3, [0.0, 0.0, 12.0])

    ordered = sort_by_depth([far, near, middle], camera)

    assert [particle["id"] for particle in ordered] == [1, 3, 2]

    first_tie = _particle(90, [0.0, 0.0, 12.0])
    second_tie = _particle(10, [0.0, 0.0, 12.0])
    third_tie = _particle(50, [0.0, 0.0, 12.0])
    tied = [first_tie, second_tie, third_tie]

    assert sort_by_depth(tied, camera) == tied
    assert sort_by_depth(tied, camera) == sort_by_depth(tied, camera)


def test_spawn_deterministic_replays_same_seed_and_index():
    first = spawn_deterministic(20261119, 7)
    second = spawn_deterministic(20261119, 7)

    assert first == second
    assert first["pos"] == [0.0, 0.0, 0.0]
    assert first["vel"] != [0.0, 0.0, 0.0]
    assert spawn_deterministic(20261119, 8)["vel"] != first["vel"]
    assert spawn_deterministic(20261120, 7)["vel"] != first["vel"]


def test_budget_limit_caps_estimated_overdraw():
    particles = [
        _particle(1, [0.0, 0.0, 0.0], radius=2.0),
        _particle(2, [1.0, 0.0, 0.0], radius=3.0),
        _particle(3, [2.0, 0.0, 0.0], radius=1.0),
        _particle(4, [3.0, 0.0, 0.0], radius=2.0),
    ]

    limited = budget_limit(particles, 16.0)

    assert [particle["id"] for particle in limited] == [1, 4]
    assert sum(particle["radius"] ** 3 for particle in limited) <= 16.0
    assert [particle["id"] for particle in particles] == [1, 2, 3, 4]


def test_lifetime_curve_hits_zero_at_end():
    for curve in ("linear", "quad", "ease"):
        assert lifetime_curve(0.0, 2.0, curve) == 1.0
        assert lifetime_curve(2.0, 2.0, curve) == 0.0
        assert lifetime_curve(2.5, 2.0, curve) == 0.0

        samples = [lifetime_curve(t, 2.0, curve) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
        assert all(
            later <= earlier + 1e-12
            for earlier, later in zip(samples, samples[1:])
        )

    assert lifetime_curve(0.5, 2.0, "linear") == pytest.approx(0.75, abs=1e-6)
    assert lifetime_curve(0.5, 2.0, "quad") == pytest.approx(0.5625, abs=1e-6)
    assert lifetime_curve(0.5, 2.0, "ease") == pytest.approx(0.84375, abs=1e-6)


def test_simulate_batch_uses_independent_pools_per_config():
    frames = [0.5, 0.5, 0.5, 0.5]
    first = _config(rate=8.0, life=2.0, capacity=20, seed=11)
    second = _config(rate=3.0, life=1.0, capacity=20, seed=22)

    combined = simulate_batch(frames, [first, second])
    first_alone = simulate_batch(frames, [first])
    second_alone = simulate_batch(frames, [second])

    assert len(combined["effects"]) == 2
    assert (
        combined["effects"][0]["frames"]
        == first_alone["effects"][0]["frames"]
    )
    assert (
        combined["effects"][1]["frames"]
        == second_alone["effects"][0]["frames"]
    )


def test_emit_rate_zero_emits_nothing():
    config = _config(rate=0.0, seed=99)
    state = {"accumulator": 0.37, "next_id": 9}
    before = copy.deepcopy(state)

    assert emit(state, 1.0, config) == []
    assert state == before


def test_emit_dt_zero_leaves_state_unchanged():
    config = _config(rate=60.0, seed=100)
    state = {"accumulator": 0.6, "next_id": 3}
    before = copy.deepcopy(state)
    particle = _particle(1, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], age=0.5)

    assert emit(state, 0.0, config) == []
    assert state == before
    assert integrate([particle], 0.0, [0.0, -9.8, 0.0], 1.0) == [particle]


def test_simulate_batch_same_seed_is_frame_identical():
    config = _config(rate=20.0, life=2.0, capacity=64, seed=424242)
    frames = [0.05] * 12

    first = simulate_batch(frames, [config])
    second = simulate_batch(frames, [config])

    assert first == second
    assert first["effects"][0]["emitted"] > 0
    assert any(first["effects"][0]["frames"])


def test_emit_capacity_zero_raises_particle_error():
    config = _config(capacity=0)
    state = {"accumulator": 0.0, "next_id": 0}
    before = copy.deepcopy(state)

    with pytest.raises(ParticleError):
        emit(state, 0.1, config)
    with pytest.raises(ParticleError):
        simulate_batch([0.1], [config])

    assert state == before


def test_simulate_batch_preserves_config_order_on_success():
    first = _config(rate=0.0, life=2.0, capacity=4, seed=1)
    second = _config(rate=10.0, life=2.0, capacity=4, seed=2)
    configs = [first, second]
    before = copy.deepcopy(configs)

    result = simulate_batch([1.0], configs)
    effects = result["effects"]

    assert [effect["config"]["rate"] for effect in effects] == [0.0, 10.0]
    assert effects[0]["frames"][0] == []
    assert len(effects[1]["frames"][0]) == 4
    assert configs == before
