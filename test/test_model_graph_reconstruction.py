import unittest

from src.model.graph_reconstruction import GraphReconstruction
from src.payload.event import Event


class TestGraphReconstruction(unittest.TestCase):
    def test_reconstruct_init(self):
        init_event = Event.from_list(*[
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 1),
        ])

        graph = GraphReconstruction(init_batch=init_event)

        inner_graph = graph.graph

        self.assertEqual(len(inner_graph.nodes), 4)
        self.assertEqual(len(inner_graph.edges), 4)

    def test_duplicate_event(self):
        init_event = Event.from_list(*[
            (1, 2),
            (2, 3),
            (3, 2),
            (1, 2),
        ])

        graph = GraphReconstruction(init_batch=init_event)

        inner_graph = graph.graph

        self.assertEqual(len(inner_graph.nodes), 3)
        self.assertEqual(len(inner_graph.edges), 2)

    def test_multi_update(self):
        graph = GraphReconstruction(init_batch=[])

        batch1 = Event.from_list(*[
            (1, 2),
            (2, 3),
            (3, 2),
            (1, 2),
        ])

        batch2 = Event.from_list(*[
            (1, 2),
            (5, 3),
            (3, 5),
            (5, 2),
        ])

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
