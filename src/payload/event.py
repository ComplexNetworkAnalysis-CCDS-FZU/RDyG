from enum import Enum
from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T", str, int)


class EventKind(Enum):
    ADD = 1
    REMOVE = 2


class Event(BaseModel, Generic[T]):
    src_node: T
    dst_node: T
    ty: EventKind = Field(EventKind.ADD)
    timestamp: Optional[int] = Field(None)
