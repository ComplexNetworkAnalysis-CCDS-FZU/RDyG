import abc
from typing import Dict, Generic, List, TypeVar, Union
import numpy as np
from torch.utils.data import Dataset

from src.payload.event_stream import EventStream
from src.payload.event import Event
from src.utils.dataloader import Data
from DyGLib.utils.utils import Data as DyGData

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
    
    @abc.abstractmethod
    def __len__(self)->int:
        pass


class BatchedDataset(Dataset, Generic[ES]):
    def __init__(self, batchtifier: BaseBatchtifier[ES]):
        self.batchtifier = batchtifier

    def __getitem__(self, index) -> Dict[str,List]:
        data = Data.from_batch(self.batchtifier[index])
        return {
            "src_node_ids":data.src_node_ids,
           "dst_node_ids": data.dst_node_ids,
            "node_interactive_time":data.node_interactive_time,
            "edge_ids":data.edge_ids,
            "label":data.label,
        }
        
        
    def __len__(self):
        return self.batchtifier.__len__()
