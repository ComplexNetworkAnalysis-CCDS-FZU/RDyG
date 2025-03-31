from enum import Enum
from typing import Optional, Tuple
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
    def from_list(cls, *data: Tuple[int, int]):
        return [
            cls(
                src_node=d[0],
                dst_node=d[1],
                ty=EventKind.ADD,
                timestamp=None,
                event_id=None,
                label=None,
            )
            for d in data
        ]
        
    def __hash__(self) -> int:
        return self.src_node.__hash__() + self.dst_node.__hash__() + self.ty.__hash__() + self.timestamp.__hash__() + self.event_id.__hash__() + self.label.__hash__()
    
    def __eq__(self, value) -> bool:
        return self.src_node == value.src_node and self.dst_node ==value.dst_node and self.timestamp == value.timestamp and self.event_id == value.event_id and self.label == value.label and self.ty == value .ty
