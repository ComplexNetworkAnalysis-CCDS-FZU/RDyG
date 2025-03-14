from functools import cached_property
from typing import Generic, List, Set, TypeVar
from more_itertools import chunked, take
import numpy as np
from pydantic import BaseModel, computed_field
import torch

from src.model.batchtifier import EventStream
from src.payload.event import Event

class Data(BaseModel):
    src_node_ids: np.ndarray
    dst_node_ids: np.ndarray
    node_interactive_time: np.ndarray
    edge_ids: np.ndarray
    label: np.ndarray

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
            [
                (e.src_node, e.dst_node, e.timestamp, e.event_id, e.label)
                for e in event
            ]
        )
        data = Data(
            src_node_ids=np.array(src, dtype=np.longlong),
            dst_node_ids=np.array(dst, dtype=np.longlong),
            node_interactive_time=np.array(timestamp, dtype=np.float64),
            edge_ids=np.array(edge_id, dtype=np.longlong),
            label=np.array(label, dtype=np.int64),
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
        if self.idx <= len(self.payload):
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



