"""Per-link delays: a swarm's links are not uniform, and the bench must be able to say so.

Several stages can share a house, where the hop is a LAN round-trip, while the boundary
between houses crosses the WAN. Emulating that needs a delay per link. These tests pin
the parsing and the indexing, because getting the index off by one would silently move
the WAN hop somewhere else and quietly invalidate every number measured with it.
"""

from __future__ import annotations

import argparse
import unittest

from distributed_runtime.benchmark import delay_for_link, link_delays


class LinkDelayParsingTests(unittest.TestCase):
    def test_single_value_parses_to_one_entry(self) -> None:
        self.assertEqual(link_delays("30"), (30.0,))

    def test_list_parses_in_order(self) -> None:
        self.assertEqual(link_delays("30,0.5,0.5,30"), (30.0, 0.5, 0.5, 30.0))

    def test_whitespace_and_trailing_separators_are_tolerated(self) -> None:
        self.assertEqual(link_delays(" 30 , 0.5 ,"), (30.0, 0.5))

    def test_zero_is_allowed(self) -> None:
        self.assertEqual(link_delays("0,0"), (0.0, 0.0))

    def test_empty_is_refused(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            link_delays(",")

    def test_negative_is_refused(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            link_delays("30,-1")


class LinkDelayIndexingTests(unittest.TestCase):
    def test_single_value_covers_every_link(self) -> None:
        delays = link_delays("30")
        self.assertEqual([delay_for_link(delays, i) for i in range(5)], [30.0] * 5)

    def test_list_is_indexed_positionally(self) -> None:
        """Link 0 is root to first child; the rest follow the chain in order."""
        delays = link_delays("30,0.5,0.5,30")
        self.assertEqual([delay_for_link(delays, i) for i in range(4)], [30.0, 0.5, 0.5, 30.0])

    def test_a_list_shorter_than_the_pipeline_is_an_error_not_a_silent_wrap(self) -> None:
        """Wrapping or defaulting here would move the WAN hop and corrupt the measurement."""
        delays = link_delays("30,0.5")
        with self.assertRaises(ValueError):
            delay_for_link(delays, 2)


if __name__ == "__main__":
    unittest.main()
