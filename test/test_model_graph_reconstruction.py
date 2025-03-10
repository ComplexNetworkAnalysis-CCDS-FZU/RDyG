import unittest

from src.model.graph_reconstruction import GraphReconstruction
from src.payload.event import Event


class TestGraphReconstruction(unittest.TestCase):
    def test_reconstruct_init(self):
        init_event = [
            Event(src_node=1, dst_node=2),
            Event(src_node=2, dst_node=3),
            Event(src_node=3, dst_node=4),
            Event(dst_node=3, src_node=1),
        ]

        graph = GraphReconstruction(init_batch=init_event)

        inner_graph = graph.graph

        self.assertEqual(len(inner_graph.nodes), 4)
        self.assertEqual(len(inner_graph.edges), 4)

    def test_duplicate_event(self):
        init_event = [
            Event(src_node=1, dst_node=2),
            Event(src_node=2, dst_node=3),
            Event(src_node=3, dst_node=2),
            Event(dst_node=1, src_node=2),
        ]

        graph = GraphReconstruction(init_batch=init_event)

        inner_graph = graph.graph

        self.assertEqual(len(inner_graph.nodes), 3)
        self.assertEqual(len(inner_graph.edges), 2)

    def test_multi_update(self):
        graph = GraphReconstruction(init_batch=[])

        batch1 = [
            Event(src_node=1, dst_node=2),
            Event(src_node=2, dst_node=3),
            Event(src_node=3, dst_node=2),
            Event(dst_node=1, src_node=2),
        ]

        batch2 = [
            Event(src_node=1, dst_node=2),
            Event(src_node=5, dst_node=3),
            Event(src_node=3, dst_node=5),
            Event(dst_node=5, src_node=2),
        ]

        # first update
        graph.update_batch(batch1)
        inner_graph = graph.graph

        self.assertEqual(len(inner_graph.nodes), 3)
        self.assertEqual(len(inner_graph.edges), 2)

        # second update
        graph.update_batch(batch2)
        inner_graph = graph.graph

        self.assertEqual(len(inner_graph.nodes), 4)
        self.assertEqual(len(inner_graph.edges), 4)
