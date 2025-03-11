import abc
from typing import Generic, List, TypeVar

from more_itertools import chunked, take
import torch

from src.payload.event import Event


class EventStream(abc.ABC):
    """实现的能获得消息流的事件流"""

    @abc.abstractmethod
    def __next__(self) -> Event:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self):
        return self


class BatchContainer:
    def __init__(self, batch: List[Event]):
        self._events = batch

    def __len__(self):
        return len(self._events)

    def __iter__(self):
        return self

    def __next__(self):
        return self._events.__next__()

    def interactive_mat(self):
        src_num = len(set(map(lambda x: x.src_node, self._events)))
        dst_num = len(set(map(lambda x: x.dst_node, self._events)))
        interactive_matrix = torch.zeros([src_num, dst_num])
        src_node_mapping = {}
        dst_node_mapping = {}

        for event in self._events:
            if event.src_node not in src_node_mapping.keys():
                src_node_mapping[event.src_node] = len(src_node_mapping)

            src_idx = src_node_mapping[event.src_node]

            if event.dst_node not in dst_node_mapping.keys():
                dst_node_mapping[event.dst_node] = len(dst_node_mapping)

            dst_idx = src_node_mapping[event.src_node]

            interactive_matrix[src_idx, dst_idx] += 1

        return interactive_matrix, src_node_mapping, dst_node_mapping

    @property
    def event(self):
        return self._events


ES = TypeVar("ES", bound=EventStream)


class Batchtifier(Generic[ES]):
    def __init__(self, event_stream: ES, batch_size: int = 100):
        self.intro = []
        self.provide_intro = False

        if len(event_stream) % batch_size != 0:
            self.intro = take(len(event_stream) % batch_size, event_stream)
            self.provide_intro = True

        self.inner_iter = iter(chunked(event_stream, batch_size))

    def __next__(self) -> BatchContainer:
        try:
            if self.provide_intro:
                self.provide_intro = False
                return self.intro
            else:
                return BatchContainer(next(self.inner_iter))
        except StopIteration:
            raise

    def __iter__(self):
        return self
