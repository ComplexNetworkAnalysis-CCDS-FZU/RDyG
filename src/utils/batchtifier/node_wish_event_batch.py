from functools import reduce
from typing import Dict, Generic, List

from src.payload.event import Event, EventKind
from src.utils.batchtifier import ES, BaseBatchtifier


class NodeWishEventBatchtifier(BaseBatchtifier, Generic[ES]):
    """节点层面的批次化结构
        - 每个节点的交互按照固定次数进行划分
        
        > 有潜在的时间不同步问题
        > - 如果在时间切分下进行定点前向切分批次化？
    """
    def __init__(self, event_stream: ES, batch_size: int = 100):
        super().__init__(event_stream)
        self.batch_size = batch_size

        self.batches: Dict[str, List[List[Event]]] = {}
        self.idx = 0

        for event in self.event_stream:
            self.add_event(event.src_node, event)
            self.add_event(event.dst_node, event)
            
        self.max_len = self.align_batch(self.batches)

    def add_event(self, encounter: int, event: Event):
        el = [event]
        key = str(encounter)

        if key not in self.batches.keys():
            self.batches[key] = [el]
        else:
            batch = self.batches[key]
            if batch[-1].__len__() < self.batch_size:
                batch[-1].extend(el)
            else:
                batch.append(el)
            self.batches[key] = batch
            
    def align_batch(self,batches:Dict[str,List[List[Event]]]):
        max_len = max(*[len(x) for x in batches.values()])
        
        def padding(l:List[List[Event]],target_len:int):
            current_len = len(l)
            padding = [Event(src_node=0,dst_node=0,ty=EventKind.ADD,timestamp=0,event_id=0,label=0)]*self.batch_size
            while current_len < target_len:
                l.insert(0,padding)
            return l
        
        # 对齐
        for k,v in batches.items():
            value = padding(v,max_len)
            batches[k] = value
        
        return max_len
        

    def __next__(self) -> List[Event]:
        if self.idx < self.max_len:
            data = reduce(lambda x,y: x + y, [v[self.idx] for  v in self.batches.values()],[])
            self.idx += 1
            return data
        else:
            raise StopIteration

    def __getitem__(self, index) -> List[Event]:
        return reduce(lambda x,y: x + y, [v[self.idx] for  v in self.batches.values()],[])
