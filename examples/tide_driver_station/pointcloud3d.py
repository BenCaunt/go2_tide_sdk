from __future__ import annotations

from typing import Optional

from tide.models.common import TideMessage, Header
from pydantic import Field


class PointCloud3D(TideMessage):
    """
    Compact 3D point cloud message.

    Stores packed XYZ float32 array and optional packed RGB uint8 array.
    - xyz: bytes of length count*3*4 (float32 x,y,z)
    - rgb: optional bytes of length count*3 (uint8 r,g,b)
    - count: number of points
    """
    header: Header = Field(default_factory=Header)
    count: int
    xyz: bytes
    rgb: Optional[bytes] = None
