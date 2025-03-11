import abc
from typing import List


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
