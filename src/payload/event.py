from enum import Enum
from typing import List, Optional, Tuple
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

    @classmethod
    def from_list(cls,*data:Tuple[int,int]):
        return [cls(src_node=d[0],dst_node=d[1],ty=EventKind.ADD,timestamp=None,event_id=None,label=None) for d in data]