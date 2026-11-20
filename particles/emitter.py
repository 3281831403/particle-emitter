"""Deterministic, frame-rate-independent particle emitter primitives.

Particles are plain mappings with the fields::

    {"id": int, "pos": [x, y, z], "vel": [vx, vy, vz],
     "age": float, "life": float, "radius": float}

``pos`` is a world-space position, ``vel`` is in world units per second,
``age`` is elapsed time in seconds, ``life`` is the total lifetime in seconds,
and ``radius`` is the world-space particle radius used for overdraw volume.
Positions and other floating-point results are compared with an absolute
tolerance of ``1e-6``.  Every time value is in seconds.

An emitter config is a mapping with ``rate`` (particles per second, >= 0),
``life`` (total lifetime in seconds, > 0), ``capacity`` (maximum retained
particles, a positive integer), and ``seed`` (an integer).  No function uses
global random state, sleeps, or depends on wall-clock time.

This module only needs the Python standard library.
"""

from __future__ import annotations

import math

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


class ParticleError(ValueError):
    """Raised when particle-emitter input is invalid or a batch is rejected."""


_MASK64 = (1 << 64) - 1
# Tolerance that absorbs binary floating-point rounding when accumulating
# fractional emission debt (e.g. 90 particles/second at 1/60 s frames).
_ACCUM_EPS = 1e-9


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_vec3(value, what: str) -> list[float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(_is_finite_number(component) for component in value)
    ):
        raise ParticleError(f"{what} must be a finite three-component vector")
    return [float(component) for component in value]


def _validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ParticleError("config must be a mapping")
    rate = config.get("rate")
    if not _is_finite_number(rate) or rate < 0:
        raise ParticleError("config rate must be finite and >= 0")
    life = config.get("life")
    if not _is_finite_number(life) or life <= 0:
        raise ParticleError("config life must be finite and > 0")
    capacity = config.get("capacity")
    if not _is_int(capacity) or capacity <= 0:
        raise ParticleError("config capacity must be a positive integer")
    seed = config.get("seed")
    if not _is_int(seed):
        raise ParticleError("config seed must be an integer")


def _validate_particle(particle: dict, require_radius: bool = False) -> None:
    if not isinstance(particle, dict):
        raise ParticleError("particle must be a mapping")
    _require_vec3(particle.get("pos"), "particle pos")
    _require_vec3(particle.get("vel"), "particle vel")
    age = particle.get("age")
    if not _is_finite_number(age) or age < 0:
        raise ParticleError("particle age must be finite and >= 0")
    life = particle.get("life")
    if not _is_finite_number(life) or life <= 0:
        raise ParticleError("particle life must be finite and > 0")
    if require_radius:
        radius = particle.get("radius")
        if not _is_finite_number(radius) or radius <= 0:
            raise ParticleError("particle radius must be finite and > 0")


def emit(state: dict, dt: float, config: dict) -> list[dict]:
    """Emit new particles for one frame and carry the fractional remainder.

    ``state`` is mutated in place and stores the emission cursor::

        {"accumulator": float, "next_id": int}

    ``accumulator`` is the fractional particle debt carried between calls and
    ``next_id`` is the index used for the next deterministic spawn.  ``dt`` is
    the frame duration in seconds and must be finite and non-negative.
    ``config`` supplies ``rate`` (particles/second, >= 0), ``life`` (total
    lifetime in seconds, > 0), ``capacity`` (positive integer) and ``seed``
    (integer).  ``capacity == 0`` raises :class:`ParticleError`.

    The call adds ``rate * dt`` to the accumulator, emits
    ``floor(accumulator)`` particles, and leaves the new fractional remainder
    in ``state`` for the next frame.  Consequently a rate below one particle
    per frame still emits over time, and a fixed rate produces the same count
    at 60 fps as at any other frame rate (up to the retained fraction).  At
    most ``capacity`` particles are returned by a single call.

    Each returned particle has ``age == 0.0``, ``life == config["life"]``,
    ``radius == 1.0``, a monotonically increasing ``id``, and a position and
    velocity from :func:`spawn_deterministic` with the config seed.  The input
    list is new; ``state`` is the only mutated argument.

    ``dt == 0`` or ``rate == 0`` returns ``[]`` without changing ``state``.
    Negative or non-finite rates, non-positive lifetimes, non-positive
    capacities, malformed state, and non-finite ``dt`` raise
    :class:`ParticleError`.  Positions are compared with tolerance ``1e-6``.
    """

    if not _is_finite_number(dt) or dt < 0:
        raise ParticleError("dt must be finite and >= 0")
    _validate_config(config)
    if not isinstance(state, dict):
        raise ParticleError("state must be a mapping")
    accumulator = state.get("accumulator")
    if not _is_finite_number(accumulator) or accumulator < 0:
        raise ParticleError("state accumulator must be finite and >= 0")
    next_id = state.get("next_id")
    if not _is_int(next_id) or next_id < 0:
        raise ParticleError("state next_id must be a non-negative integer")

    rate = float(config["rate"])
    if dt == 0 or rate == 0:
        return []

    total = accumulator + rate * dt
    count = int(math.floor(total + _ACCUM_EPS))
    remainder = total - count
    if remainder < 0.0:
        # Only reachable through floating-point rounding at an integer boundary.
        remainder = 0.0
    count = min(count, config["capacity"])

    seed = config["seed"]
    life = float(config["life"])
    spawned = []
    for offset in range(count):
        index = next_id + offset
        spawn = spawn_deterministic(seed, index)
        spawned.append(
            {
                "id": index,
                "pos": spawn["pos"],
                "vel": spawn["vel"],
                "age": 0.0,
                "life": life,
                "radius": 1.0,
            }
        )

    state["accumulator"] = remainder
    state["next_id"] = next_id + count
    return spawned


def integrate(
    particles: list[dict],
    dt: float,
    gravity: list[float],
    drag: float,
) -> list[dict]:
    """Advance particles by ``dt`` seconds using semi-implicit Euler and drag.

    The input particles are not mutated; the returned list contains copies.
    ``gravity`` is a three-component acceleration in world units per second
    squared.  ``drag`` is a non-negative exponential damping coefficient in
    1/seconds.  For every particle the update order is::

        vel += gravity * dt
        vel *= exp(-drag * dt)
        pos += vel * dt

    Position therefore uses the already-updated velocity (semi-implicit Euler)
    instead of the previous velocity.  Drag is applied continuously per second,
    so the same elapsed time produces the same velocity loss regardless of how
    many frames were used.

    ``age`` increases by ``dt``; a particle whose ``age >= life`` is removed
    from the result.  Times are in seconds.  ``dt == 0`` returns an unchanged
    copy.  Negative or non-finite ``dt``, a negative or non-finite ``drag``, a
    malformed gravity vector, or a malformed particle raises
    :class:`ParticleError`.  Positions and velocities are compared with
    tolerance ``1e-6``.
    """

    if not _is_finite_number(dt) or dt < 0:
        raise ParticleError("dt must be finite and >= 0")
    if not _is_finite_number(drag) or drag < 0:
        raise ParticleError("drag must be finite and >= 0")
    gravity = _require_vec3(gravity, "gravity")
    for particle in particles:
        _validate_particle(particle)

    damping = math.exp(-drag * dt)
    updated = []
    for particle in particles:
        age = particle["age"] + dt
        if age >= particle["life"]:
            continue
        vel = [
            (particle["vel"][axis] + gravity[axis] * dt) * damping
            for axis in range(3)
        ]
        pos = [particle["pos"][axis] + vel[axis] * dt for axis in range(3)]
        updated.append(
            {
                "id": particle["id"],
                "pos": pos,
                "vel": vel,
                "age": age,
                "life": particle["life"],
                "radius": particle.get("radius", 1.0),
            }
        )
    return updated


def recycle(particles: list[dict], capacity: int) -> list[dict]:
    """Keep at most ``capacity`` particles, dropping the oldest ones.

    ``particles`` is ordered oldest-first.  When it contains more than
    ``capacity`` items, the result keeps the last ``capacity`` (newest)
    particles in their original order; new particles are never the ones
    discarded.  ``capacity == 0`` returns ``[]``.  The input list is not
    mutated and the returned list may contain the original particle mappings.

    ``capacity`` must be a non-negative integer; a negative or non-integer
    value raises :class:`ParticleError`.  This is the hard capacity limit for
    every particle pool.
    """

    if not _is_int(capacity) or capacity < 0:
        raise ParticleError("capacity must be a non-negative integer")
    if capacity == 0:
        return []
    if len(particles) <= capacity:
        return list(particles)
    return list(particles[-capacity:])


def sort_by_depth(particles: list[dict], camera: dict) -> list[dict]:
    """Return particles sorted far-to-near for correct alpha blending.

    Depth is the Euclidean distance from ``camera["pos"]`` to
    ``particle["pos"]``.  Transparent particles are blended back-to-front, so
    the greatest distance comes first.  Equal depths preserve their input
    order (the sort is stable) rather than depending on object identity or
    memory address.

    The input sequence and mappings are not mutated.  ``camera`` must contain
    a finite three-component ``pos``; malformed particles or cameras raise
    :class:`ParticleError`.  Positions are compared with tolerance ``1e-6``.
    """

    if not isinstance(camera, dict):
        raise ParticleError("camera must be a mapping")
    camera_pos = _require_vec3(camera.get("pos"), "camera pos")
    for particle in particles:
        _validate_particle(particle)

    def depth(particle: dict) -> float:
        pos = particle["pos"]
        return math.sqrt(
            sum((pos[axis] - camera_pos[axis]) ** 2 for axis in range(3))
        )

    # ``sorted`` is stable, so equal depths keep their input order.
    return sorted(particles, key=depth, reverse=True)


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _hash_to_unit(value: int) -> float:
    # Top 53 bits mapped into [-1, 1).
    return ((value >> 11) / float(1 << 53)) * 2.0 - 1.0


def spawn_deterministic(seed: int, index: int) -> dict:
    """Build the deterministic spawn data for ``seed`` and ``index``.

    The result is ``{"pos": [0.0, 0.0, 0.0], "vel": [vx, vy, vz]}``.  The
    velocity is finite and non-zero and is derived only from ``seed`` and
    ``index`` with a stable hash, never from global random state.  The same
    pair therefore produces an exactly equal result on every call and in every
    process, while different indices normally produce different velocities.

    ``seed`` must be an integer and ``index`` must be a non-negative integer;
    other input raises :class:`ParticleError`.  Positions are compared with
    tolerance ``1e-6``.
    """

    if not _is_int(seed):
        raise ParticleError("seed must be an integer")
    if not _is_int(index) or index < 0:
        raise ParticleError("index must be a non-negative integer")

    # Odd multipliers are bijective mod 2**64, so distinct (seed, index)
    # pairs never collapse onto the same hash input.
    base = (
        (seed & _MASK64) * 0xA24BAED4963EE407
        + index * 0x9FB21C651E98DF25
    ) & _MASK64
    h1 = _splitmix64(base)
    h2 = _splitmix64(h1)
    h3 = _splitmix64(h2)
    vel = [_hash_to_unit(h1), _hash_to_unit(h2), _hash_to_unit(h3)]
    if vel == [0.0, 0.0, 0.0]:
        vel[0] = 1.0
    return {"pos": [0.0, 0.0, 0.0], "vel": vel}


def budget_limit(particles: list[dict], max_overdraw: float) -> list[dict]:
    """Trim particles so estimated total overdraw stays within budget.

    A particle's estimated overdraw volume is ``radius ** 3`` (the particle
    radius is in world units).  The total overdraw of a list is the sum of
    those volumes.  Candidates are considered from largest volume to smallest,
    keeping a candidate only when it fits the remaining budget; ties keep
    input order.  The survivors are returned in their original input order.
    This caps dense bursts instead of letting every particle draw.

    ``max_overdraw`` is the non-negative volume budget; ``0`` returns ``[]``.
    The input list is not mutated.  A negative or non-finite budget or a
    non-positive/non-finite particle radius raises :class:`ParticleError`.
    Positions are compared with tolerance ``1e-6``.
    """

    if not _is_finite_number(max_overdraw) or max_overdraw < 0:
        raise ParticleError("max_overdraw must be finite and >= 0")
    for particle in particles:
        _validate_particle(particle, require_radius=True)

    if max_overdraw == 0:
        return []

    indexed = list(enumerate(particles))
    # Largest volume first; the stable sort keeps input order among ties.
    indexed.sort(key=lambda item: -(item[1]["radius"] ** 3))

    remaining = max_overdraw
    kept_indices = []
    for original_index, particle in indexed:
        volume = particle["radius"] ** 3
        if volume <= remaining + _ACCUM_EPS:
            remaining -= volume
            kept_indices.append(original_index)

    kept_indices.sort()
    return [particles[index] for index in kept_indices]


def lifetime_curve(t: float, life: float, curve: str) -> float:
    """Return normalized lifetime opacity in the closed interval ``[0, 1]``.

    ``t`` is elapsed time in seconds and ``life`` is the total lifetime in
    seconds (``life > 0``).  With ``x = clamp(t / life, 0.0, 1.0)`` the curves
    are:

    * ``"linear"``: ``1 - x``
    * ``"quad"``: ``(1 - x) ** 2``
    * ``"ease"``: ``1 - x ** 2 * (3 - 2 * x)``

    At ``t == 0`` the result is exactly ``1.0``; at ``t >= life`` it is
    exactly ``0.0`` so a disappearing particle never flashes back on.  The
    result is monotone non-increasing in ``t``.  Unknown curve names,
    non-positive lives, and non-finite ``t`` raise :class:`ParticleError`.
    Values are compared with tolerance ``1e-6``.
    """

    if curve not in ("linear", "quad", "ease"):
        raise ParticleError(f"unknown lifetime curve: {curve!r}")
    if not _is_finite_number(life) or life <= 0:
        raise ParticleError("life must be finite and > 0")
    if not _is_finite_number(t):
        raise ParticleError("t must be finite")

    x = min(max(t / life, 0.0), 1.0)
    if curve == "linear":
        return 1.0 - x
    if curve == "quad":
        return (1.0 - x) ** 2
    return 1.0 - x * x * (3.0 - 2.0 * x)


_BATCH_GRAVITY = [0.0, -9.8, 0.0]
_BATCH_DRAG = 0.0


def simulate_batch(frames: list[float], configs: list[dict]) -> dict:
    """Simulate several emitters independently over a sequence of frames.

    ``frames`` is a sequence of non-negative frame durations in seconds.
    ``configs`` is a sequence of emitter config mappings with ``rate``,
    ``life``, ``capacity`` and ``seed`` following the same contract as
    :func:`emit`.

    Every config owns a separate emission state and particle pool for the
    whole batch; pools are never shared between configs.  For each frame, the
    configs are processed in input order: emit, integrate with fixed gravity
    and drag, then recycle to ``capacity``.  Each stored frame is a deep copy,
    so later integration cannot rewrite earlier frames.

    The return value is::

        {"effects": [
            {"config": <deep copy of config>,
             "frames": [[particle, ...], ...],
             "emitted": <total emitted for this config>},
            ...
        ]}

    with one effect per config in the original input order and one frame per
    entry in ``frames``.  All frames and configs are validated before any
    simulation starts.  If any value is invalid (including ``capacity == 0``),
    the whole call raises :class:`ParticleError` and returns no partial
    result; input mappings are not modified.  Equal inputs, including equal
    seeds, produce equal results frame by frame.  Positions are compared with
    tolerance ``1e-6``.
    """

    # Validate everything up front so an invalid entry rejects the whole
    # batch before any simulation work happens.
    if not isinstance(frames, (list, tuple)):
        raise ParticleError("frames must be a sequence of durations")
    for dt in frames:
        if not _is_finite_number(dt) or dt < 0:
            raise ParticleError("frame durations must be finite and >= 0")
    if not isinstance(configs, (list, tuple)):
        raise ParticleError("configs must be a sequence of mappings")
    for config in configs:
        _validate_config(config)

    effects = []
    for config in configs:
        state = {"accumulator": 0.0, "next_id": 0}
        pool: list[dict] = []
        snapshots = []
        emitted = 0
        for dt in frames:
            spawned = emit(state, dt, config)
            emitted += len(spawned)
            pool = integrate(pool + spawned, dt, _BATCH_GRAVITY, _BATCH_DRAG)
            pool = recycle(pool, config["capacity"])
            # ``integrate`` already copies, but snapshot explicitly so a
            # stored frame can never alias the live pool.
            snapshots.append(
                [
                    {
                        "id": particle["id"],
                        "pos": list(particle["pos"]),
                        "vel": list(particle["vel"]),
                        "age": particle["age"],
                        "life": particle["life"],
                        "radius": particle["radius"],
                    }
                    for particle in pool
                ]
            )
        effects.append(
            {
                "config": dict(config),
                "frames": snapshots,
                "emitted": emitted,
            }
        )
    return {"effects": effects}
