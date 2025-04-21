from functools import cached_property
from typing import List, Set, TypeVar, Union
import numpy as np
from pydantic import BaseModel, computed_field

from src.payload.event_stream import EventStream
from src.payload.event import Event


class Data(BaseModel):
    src_node_ids: List[int]
    dst_node_ids: List[int]
    node_interactive_time: List[float]
    edge_ids: List[int]
    label: List[float]

    @computed_field
    @cached_property
    def num_interactive(self) -> int:
        return len(self.src_node_ids)

    @computed_field
    @cached_property
    def unique_node_ids(self) -> Set[int]:
        return set(self.src_node_ids) | set(self.dst_node_ids)

    @computed_field
    @cached_property
    def num_unique_nodes(self) -> int:
        return len(self.unique_node_ids)

    @staticmethod
    def from_batch(event: List[Event]):
        src, dst, timestamp, edge_id, label = zip(
            *[
                (e.src_node, e.dst_node, e.timestamp, e.event_id, e.label)
                for e in event
                if e.src_node != 0 or e.dst_node != 0
            ]
        )
        data = Data(
            src_node_ids=list(src),
            dst_node_ids=list(dst),
            node_interactive_time=list(timestamp),
            edge_ids=list(edge_id),
            label=list(label),
        )
        return data
    
    class Config:
        arbitrary_types_allowed = True


class SplitEventStream(EventStream):
    def __init__(self, event_list: List[Event]):
        super().__init__()
        self.payload: List[Event] = event_list
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> Event:
        if self.idx < len(self.payload):
            data = self.payload[self.idx]
            self.idx += 1
            return data
        else:
            raise StopIteration

    def __len__(self):
        return self.payload.__len__()

    @classmethod
    def spilt_event_stream_at(cls, event_stream: EventStream, at: int):
        data = list(event_stream)
        front = data[:at]
        rear = data[at:]
        return cls(front), cls(rear)


ES = TypeVar("ES", bound=EventStream)
