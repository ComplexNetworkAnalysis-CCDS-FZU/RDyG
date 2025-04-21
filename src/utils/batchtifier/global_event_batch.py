from typing import Generic, List

from more_itertools import chunked, take
from src.payload.event import Event
from src.utils.batchtifier import ES, BaseBatchtifier


class GlobalEventBatchtifier(BaseBatchtifier, Generic[ES]):
    """
    全局事件批次化器
    - 按照全部的交互事件进行批次划分
    - 按照固定事件数量划分批次

    * 这样划分可能各个批次难以对齐
    """

    def __init__(self, event_stream: ES, batch_size: int = 100):
        self.intro: List[Event] = []
        self.provide_intro = False

        if len(event_stream) % batch_size != 0:
            self.intro = take(len(event_stream) % batch_size, event_stream)
            self.provide_intro = True

        self.inner_iter: List[List[Event]] = list(chunked(event_stream, batch_size))
        self.idx = 0

    def __next__(self) -> List[Event]:
        if self.provide_intro:
            self.provide_intro = False
            return self.intro
        elif self.idx < len(self.inner_iter):
            data = self.inner_iter[self.idx]
            self.idx += 1
            return data
        else:
            raise StopIteration

    def __getitem__(self, index):
        if index == 0:
            return self.intro
        else:
            return self.inner_iter[index - 1]
        
    def __len__(self):
        return len(self.inner_iter)
