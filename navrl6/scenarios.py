"""Fixed robot start / goal pairs for the six robots.

Coordinates are transcribed from the first frame of the reference videos
(``step 1``), so every scenario uses the same initial assignment:

| Robot | Colour  | Start (m)       | Goal (m)        |
|-------|---------|-----------------|-----------------|
| 1     | blue    | (12.5,  0.5)    | (-7.2,  8.6)    |
| 2     | red     | ( 7.2, 10.2)    | (-10.5, -2.1)   |
| 3     | green   | (-7.8, 11.4)    | (-2.8, -10.2)   |
| 4     | purple  | (-13.2, 0.6)    | ( 7.8, -6.6)    |
| 5     | cyan    | ( 9.8,  5.2)    | (-6.8, -10.6)   |
| 6     | pink    | ( 6.8, -11.1)   | ( 1.8, 10.5)    |
"""

from __future__ import annotations

import numpy as np

ROBOT_STARTS: np.ndarray = np.array(
    [
        [12.5, 0.5],
        [7.2, 10.2],
        [-7.8, 11.4],
        [-13.2, 0.6],
        [9.8, 5.2],
        [6.8, -11.1],
    ],
    dtype=float,
)

ROBOT_GOALS: np.ndarray = np.array(
    [
        [-7.2, 8.6],
        [-10.5, -2.1],
        [-2.8, -10.2],
        [7.8, -6.6],
        [-6.8, -10.6],
        [1.8, 10.5],
    ],
    dtype=float,
)

SCENARIOS = ("static", "dynamic", "mixed")
