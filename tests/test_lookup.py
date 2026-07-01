# unit tests for the 4-tier lookup
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.lookup import option_set, build_lookup_tables, lookup


def _row(core_q, opts, answer):
    r = {"core_q": core_q, "answer": answer}
    for L, v in zip("ABCDE", opts):
        r[L] = v
    return r


def _df(rows):
    return pd.DataFrame(rows)


def test_option_set_is_sorted():
    r = _row("q1", ["b", "a", "d", "c", "e"], "A")
    assert option_set(r) == ("a", "b", "c", "d", "e")


def test_t1_exact_match():
    train = _df([_row("what is 2+2", ["1", "2", "3", "4", "5"], "D")])
    tables = build_lookup_tables(train)
    letter, tier = lookup(_row("what is 2+2", ["1", "2", "3", "4", "5"], "?"), tables)
    assert (letter, tier) == ("D", "T1")


def test_t2_same_options_different_core_q():
    train = _df([_row("what is 2+2", ["1", "2", "3", "4", "5"], "D")])
    tables = build_lookup_tables(train)
    letter, tier = lookup(_row("compute 2+2", ["1", "2", "3", "4", "5"], "?"), tables)
    assert (letter, tier) == ("D", "T2")


def test_t3_always_correct_text():
    train = _df([
        _row("q1", ["sun", "moon", "mars", "venus", "jupiter"], "B"),
        _row("q2", ["sun", "moon", "mars", "venus", "jupiter"], "B"),
    ])
    tables = build_lookup_tables(train)
    letter, tier = lookup(
        _row("q_novel", ["sun", "moon", "pluto", "neptune", "saturn"], "?"), tables
    )
    assert (letter, tier) == ("B", "T3")


def test_t4_core_q_known_answer():
    # 'moon' is correct once and wrong once, so it's NOT in T3.
    # But T4 still knows moon as a correct answer for q1.
    train = _df([
        _row("q1", ["a", "moon", "c", "d", "e"], "B"),
        _row("q2", ["moon", "b", "c", "d", "e"], "B"),
    ])
    tables = build_lookup_tables(train)
    letter, tier = lookup(_row("q1", ["moon", "x", "y", "z", "w"], "?"), tables)
    assert (letter, tier) == ("A", "T4")


def test_no_tier_fires():
    train = _df([_row("q1", ["a", "b", "c", "d", "e"], "A")])
    tables = build_lookup_tables(train)
    letter, tier = lookup(_row("q2", ["v", "w", "x", "y", "z"], "?"), tables)
    assert (letter, tier) == (None, None)
