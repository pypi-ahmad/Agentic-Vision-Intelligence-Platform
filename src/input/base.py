"""Input source abstraction — base types."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

import numpy as np


@dataclass
class FramePacket:
    """A single frame with metadata."""
    frame: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    frame_index: int = 0
    source_id: str = ""
    source_type: str = ""  # "camera", "image", "video"
    metadata: dict = field(default_factory=dict)


class InputSource(abc.ABC):
    """Abstract base class for all input sources."""
    source_type: str = "base"

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def read(self) -> FramePacket | None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    def frames(self, sample_rate: int = 1) -> Iterator[FramePacket]:
        self.open()
        try:
            idx = 0
            while True:
                pkt = self.read()
                if pkt is None:
                    break
                if idx % sample_rate == 0:
                    pkt.frame_index = idx
                    yield pkt
                idx += 1
        finally:
            self.close()

    @property
    def is_live(self) -> bool:
        return False
