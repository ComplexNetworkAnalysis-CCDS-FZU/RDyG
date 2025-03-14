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

