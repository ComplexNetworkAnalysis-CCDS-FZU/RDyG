import abc
from typing import Generic, List, TypeVar

from more_itertools import chunked, take

from src.payload.event import Event


class EvenStream(abc.ABC):
    """实现的能获得消息流的事件流"""

    @abc.abstractmethod
    def __next__(self) -> Event:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __iter__(self):
        return self


ES = TypeVar("ES", bound=EvenStream)


class Batchtifier(Generic[ES]):
    def __init__(self, event_stream: ES, batch_size: int = 100):
        self.intro = []
        self.provide_intro = False

        if len(event_stream) % batch_size != 0:
            self.intro = take(len(event_stream) % batch_size, event_stream)
            self.provide_intro = True

        self.inner_iter = iter(chunked(event_stream, batch_size))

    def __next__(self) -> List[Event]:
        try:
            if self.provide_intro:
                self.provide_intro = False
                return self.intro
            else:
                return next(self.inner_iter)
        except StopIteration:
            raise

    def __iter__(self):
        return self
