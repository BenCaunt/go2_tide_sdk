from __future__ import annotations

from typing import List

from pydantic import Field
from tide.models.common import TideMessage, Header


class OccupancyGrid2D(TideMessage):
    """
    Simple 2D occupancy grid for mapping and planning.

    - width, height: number of cells
    - resolution: meters per cell
    - origin_x, origin_y: world coordinates (meters) of the grid's (0,0) cell
    - data: row-major flattened list of int16 values sized width*height
            -1 = unknown, 0 = free, >=100 = occupied
    """

    header: Header = Field(default_factory=Header)
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    data: List[int]

