from typing import List
from src.payload.dataset import BaseDataset, NodeMapper
from src.payload.event import Event, EventKind
from src.payload.event_stream import EventStream


class MyketEventStream(EventStream):
    def __init__(self, events: List[List[str]]) -> None:
        super().__init__()
        self.events = events
        self.idx = 0
        self.max_idx = len(events)

    def __next__(self) -> Event:
        if self.idx < self.max_idx:
            src_node, dst_node, timestamp, label, id, _ = self.events[self.idx]

            event = Event(
                src_node=int(src_node),
                dst_node=int(dst_node),
                timestamp=float(timestamp),
                event_id=int(id),
                label=float(label),
                ty=EventKind.ADD,
            )

            self.idx += 1
            return event
        else:
            raise StopIteration

    def __len__(self) -> int:
        return self.max_idx - 1


class MyketDataset(BaseDataset):
    def __init__(self, path: str = "./datasets/myket/myket.csv") -> None:
        super().__init__()
        with open(path, "r") as file:
            self.lines = [line.strip().split(",") for line in file.readlines()[1:]]

        self.stream = MyketEventStream(self.lines)

        self.mapper = NodeMapper()
        for src, dst, _, _, _, _ in self.lines:
            self.mapper.add_node(int(src))
            self.mapper.add_node(int(dst))

    def event_stream(self) -> EventStream:
        return self.stream

    def node_map(self) -> NodeMapper:
        return self.mapper

    def __len__(self):
        return self.lines.__len__()
