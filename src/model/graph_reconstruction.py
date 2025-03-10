from typing import List
from networkx import Graph

from src.payload.event import Event


class GraphReconstruction(object):
    def __init__(self, /, init_batch: List[Event]):
        self._graph = Graph()
        if len(init_batch) != 0:
            self.update_batch(init_batch)

    def update_batch(self, batch: List[Event]):
        for event in batch:
            self._graph.add_edge(event.src_node, event.dst_node)

    @property
    def graph(self):
        return self._graph
