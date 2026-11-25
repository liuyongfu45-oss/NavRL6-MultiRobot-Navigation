"""Circular obstacles: static disks and moving (dynamic) disks.

Dynamic obstacles move at constant speed with a random heading and bounce
off the arena walls, exactly like the orange disks in the reference videos.
"""

from __future__ import annotations

import numpy as np

from . import config


class Obstacle:
    """A single circular obstacle.

    Parameters
    ----------
    position : array-like, shape (2,)
        Initial ``(x, y)`` position in metres.
    radius : float
        Disk radius in metres.
    velocity : array-like or None
        Constant ``(vx, vy)`` in m/s. ``None`` means a static obstacle.
    kind : str
        ``"static"`` or ``"dynamic"``.
    """

    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        velocity: np.ndarray | None = None,
        kind: str = "static",
    ) -> None:
        self.position = np.asarray(position, dtype=float)
        self.radius = float(radius)
        self.velocity = (
            None if velocity is None else np.asarray(velocity, dtype=float)
        )
        self.kind = kind

    # ------------------------------------------------------------------ #
    @property
    def is_dynamic(self) -> bool:
        return self.kind == "dynamic"

    def update(self, dt: float, world_size: float) -> None:
        """Advance a dynamic obstacle one step; reflect off the walls."""
        if self.velocity is None:
            return
        self.position = self.position + self.velocity * dt
        for axis in (0, 1):
            if self.position[axis] > world_size - self.radius:
                self.position[axis] = world_size - self.radius
                self.velocity[axis] *= -1.0
            elif self.position[axis] < -world_size + self.radius:
                self.position[axis] = -world_size + self.radius
                self.velocity[axis] *= -1.0

    def distance_to(self, point: np.ndarray) -> float:
        """Euclidean distance from the disk centre to ``point``."""
        return float(np.linalg.norm(self.position - np.asarray(point, dtype=float)))

    def collision_with(self, point: np.ndarray, radius: float) -> bool:
        """Return ``True`` if a circle of ``radius`` at ``point`` touches this disk."""
        return self.distance_to(point) < self.radius + radius

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"Obstacle(kind={self.kind}, pos={self.position.round(2).tolist()}, "
            f"r={self.radius:.2f})"
        )


def _random_position(rng: np.random.Generator, radius: float,
                     world_size: float) -> np.ndarray:
    margin = radius + 0.5
    return rng.uniform(-world_size + margin, world_size - margin, size=2)


def build_obstacles(scenario: str, seed: int = 0) -> list[Obstacle]:
    """Create the 15 obstacles for a scenario.

    Parameters
    ----------
    scenario : {"static", "dynamic", "mixed"}
    seed : int
        Reproducible random seed.

    Returns
    -------
    list[Obstacle]
    """
    if scenario not in ("static", "dynamic", "mixed"):
        raise ValueError(f"Unknown scenario: {scenario!r}")

    rng = np.random.default_rng(seed)
    obstacles: list[Obstacle] = []

    if scenario == "static":
        n_static, n_dynamic = config.NUM_OBSTACLES, 0
    elif scenario == "dynamic":
        n_static, n_dynamic = 0, config.NUM_OBSTACLES
    else:
        n_static, n_dynamic = config.N_STATIC_MIXED, config.N_DYNAMIC_MIXED

    def _place(kind: str, radius_bounds: tuple[float, float]) -> None:
        radius = rng.uniform(*radius_bounds)
        position = _random_position(rng, radius, config.WORLD_SIZE)
        # keep a minimum separation from already placed obstacles
        for _ in range(50):
            if all(np.linalg.norm(position - o.position) >
                   radius + o.radius + 0.8 for o in obstacles):
                break
            position = _random_position(rng, radius, config.WORLD_SIZE)

        velocity = None
        if kind == "dynamic":
            heading = rng.uniform(0.0, 2.0 * np.pi)
            velocity = config.DYNAMIC_SPEED * np.array([np.cos(heading), np.sin(heading)])
        obstacles.append(Obstacle(position, radius, velocity, kind))

    for _ in range(n_static):
        _place("static", config.STATIC_RADIUS)
    for _ in range(n_dynamic):
        _place("dynamic", config.DYNAMIC_RADIUS)

    return obstacles
