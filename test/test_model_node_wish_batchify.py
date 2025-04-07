import unittest

from src.payload.event import Event
from src.payload.event_stream import EventStream
from src.utils.batchtifier.node_wish_event_batch import NodeWishEventBatchtifier


class TestEvenStream(EventStream):
    def __init__(self, *events: Event):
        super().__init__()
        self.events = events
        self.idx = 0

    def __next__(self):
        if self.idx >= len(self.events):
            raise StopIteration
        value = self.events[self.idx]
        self.idx += 1
        return value

    def __len__(self):
        return len(self.events)


class TestNodeWishBatcher(unittest.TestCase):
    def test_no_align_batch(self):
        event_stream = TestEvenStream(
            *Event.from_list((1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (1, 2))
        )

        batchtifier = NodeWishEventBatchtifier[TestEvenStream](event_stream, 2)
        batch_iter = iter(batchtifier)

        batch1 = next(batch_iter)
        self.assertEqual(3, len(batch1))

        batch2 = next(batch_iter)
        self.assertEqual(4, len(batch2))

    def test_align_batch(self):
        event_stream = TestEvenStream(*Event.from_list((2, 3), (1, 4), (2, 4), (1, 2)))

        batchtifier = NodeWishEventBatchtifier[TestEvenStream](event_stream, 2)
        batch_iter = iter(batchtifier)

        batch1 = next(batch_iter)
        print(batch1)
        self.assertEqual(4, len(batch1))

        with self.assertRaises(StopIteration):
            next(batch_iter)
