import abc
from typing import Generic, List, TypeVar
from torch.utils.data import Dataset

from src.payload.event_stream import EventStream
from src.payload.event import Event
from src.utils.dataloader import Data

ES = TypeVar("ES", bound=EventStream)


class BaseBatchtifier(abc.ABC, Generic[ES]):
    """批次划分的抽象基类, 根据提供的数据集的迭代器，构造按照批次划分"""

    def __init__(self, event_stream: ES):
        super().__init__()
        self.event_stream = event_stream

    @abc.abstractmethod
    def __next__(self) -> List[Event]:
        pass

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __getitem__(self, index) -> List[Event]:
        pass


class BatchedDataset(Dataset, Generic[ES]):
    def __init__(self, batchtifier: BaseBatchtifier[ES]):
        self.batchtifier = batchtifier

    def __getitem__(self, index) -> Data:
        return Data.from_batch(self.batchtifier[index])
