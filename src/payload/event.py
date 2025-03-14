from enum import Enum
from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, Field



class EventKind(Enum):
    ADD = 1
    REMOVE = 2


class Event(BaseModel):
    src_node: int
    dst_node: int
    ty: EventKind = Field(EventKind.ADD)
    timestamp: Optional[int] = Field(None)
    event_id: Optional[int] = Field(0)
    label: Optional[int] = Field(None)
