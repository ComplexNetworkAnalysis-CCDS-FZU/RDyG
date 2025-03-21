from typing import List
import unittest


from src.payload.event_stream import EventStream
from src.payload.event import Event
from src.utils.batchtifier.global_event_batch import (
    GlobalEventBatchtifier as Batchtifier,
)


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

    @classmethod
    def load(cls, payload: List[dict]):
        pass


class TestBatchtifier(unittest.TestCase):
    def test_full_batch(self):
        batch = Batchtifier[TestEvenStream](
            TestEvenStream(*Event.from_list((1, 2), (2, 3), (3, 1), (4, 2))),
            batch_size=2,
        )

        iterator = iter(batch)

        item = next(iterator)
        self.assertEqual(len(item), 2)

        item = next(iterator)
        self.assertEqual(len(item), 2)

        with self.assertRaises(StopIteration):
            next(iterator)

    def test_not_full_batch(self):
        """如果事件流不能刚好被划分为指定长度的批次，第一个批次将较短"""
        batch = Batchtifier[TestEvenStream](
            TestEvenStream(*Event.from_list((1, 2), (2, 3), (3, 1))),
            batch_size=2,
        )

        iterator = iter(batch)

        item = next(iterator)
        self.assertEqual(len(item), 1)

        item = next(iterator)
        self.assertEqual(len(item), 2)

        with self.assertRaises(StopIteration):
            next(iterator)

    def test_item_not_enough(self):
        batch = Batchtifier[TestEvenStream](
            TestEvenStream(*Event.from_list((1, 2), (2, 3), (3, 1))),
            batch_size=20,
        )

        iterator = iter(batch)

        item = next(iterator)
        self.assertEqual(len(item), 3)

        with self.assertRaises(StopIteration):
            next(iterator)


if __name__ == "__main__":
    unittest.main()
