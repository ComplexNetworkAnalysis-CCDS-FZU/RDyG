
import abc
from operator import length_hint
from typing import Dict, List

from src.payload.event_stream import EventStream

class NodeMapper:
    def __init__(self) -> None:
        self.node2id:Dict[int,int]={}
        self.id2node:List[int] = []
        
    def add_node(self,node:int)->int:
        if node in self.node2id.keys():
            return self.node2id[node]
        else:
            id =len(self.id2node)
            self.id2node.append(node)
            self.node2id[node] = id
            return id

    def get_node_id(self,node:int)->int:
        return self.node2id[node]
    
    def get_node(self,id:int)->int:
        return self.id2node[id]
    
    def node_number(self)->int:
        return len(self.id2node)
    
    
class BaseDataset(abc.ABC):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def event_stream(self)->EventStream:
        pass
    
    @abc.abstractmethod
    def node_map(self)->NodeMapper:
        pass
    
    
    