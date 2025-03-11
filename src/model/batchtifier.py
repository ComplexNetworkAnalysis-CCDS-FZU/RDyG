import abc
from typing import Generic, List, TypeVar

from more_itertools import chunked, take
import torch
import torch.utils
import torch.utils.data

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

    @property
    def event(self):
        return self._events


ES = TypeVar("ES", bound=EventStream)


class Batchtifier(
    torch.utils.data.Dataset,
    Generic[ES],
):
    def __init__(self, event_stream: ES, batch_size: int = 100):
        self.intro = []
        self.provide_intro = False

        if len(event_stream) % batch_size != 0:
            self.intro = take(len(event_stream) % batch_size, event_stream)
            self.provide_intro = True

        self.inner_iter = list(chunked(event_stream, batch_size))
        self.idx = 0

    def __next__(self) -> BatchContainer:
        if self.provide_intro:
            self.provide_intro = False
            return self.intro
        elif self.idx < len(self.inner_iter):
            data = BatchContainer(self.inner_iter[self.idx])
            self.idx += 1
            return data
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if index == 0:
            return self.intro
        else:
            return self.inner_iter[index - 1]
