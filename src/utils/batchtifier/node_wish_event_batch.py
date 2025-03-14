from typing import Dict, Generic, List

from src.payload.event import Event
from src.utils.batchtifier import ES, BaseBatchtifier


class NodeWishEventBatchtifier(BaseBatchtifier, Generic[ES]):
    def __init__(self, event_stream: ES, batch_size: int = 100):
        super().__init__(event_stream)
        self.batch_size = batch_size

        self.batches: Dict[str, List[List[Event]]] = {}
        self.idx = 0

        for event in self.event_stream:
            self.add_event(event.src_node, event)
            self.add_event(event.dst_node, event)

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

    def __next__(self) -> Dict[str, List[Event]]:
        data = {k: v[self.idx] for k, v in self.batches.items()}
        self.idx += 1
        return data

    def __getitem__(self, index) -> Dict[str, List[Event]]:
        return {k: v[index] for k, v in self.batches.items()}
