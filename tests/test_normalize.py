# unit tests for normalize_stem
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalize import normalize_stem


def test_strips_each_prefix():
    cases = [
        "Pick the best possible answer: What is 2 + 2?",
        "Select the most accurate option: What is 2 + 2?",
        "Determine the correct option: What is 2 + 2?",
        "Identify the correct statement: What is 2 + 2?",
        "Choose the correct answer: What is 2 + 2?",
        "Which of the following is correct? What is 2 + 2?",
    ]
    for c in cases:
        assert normalize_stem(c) == "What is 2 + 2?", c


def test_strips_each_suffix():
    cases = [
        "What is 2 + 2? among the listed options.",
        "What is 2 + 2? based on the given context.",
        "What is 2 + 2? from the following choices.",
        "What is 2 + 2? carefully.",
    ]
    for c in cases:
        assert normalize_stem(c) == "What is 2 + 2?", c


def test_strips_compound_prefix_and_suffix():
    s = "Pick the best possible answer: What is 2 + 2? among the listed options."
    assert normalize_stem(s) == "What is 2 + 2?"


def test_strips_nested_prefixes():
    s = "Pick the best possible answer: Choose the correct answer: What is 2 + 2?"
    assert normalize_stem(s) == "What is 2 + 2?"


def test_idempotent():
    s = "Pick the best possible answer: What is 2 + 2? among the listed options."
    once = normalize_stem(s)
    twice = normalize_stem(once)
    assert once == twice


def test_no_wrapper_passes_through():
    s = "What is the capital of France?"
    assert normalize_stem(s) == s


def test_whitespace_normalized():
    s = "Pick the best possible answer:   What  is   2 + 2?  among the listed options."
    assert normalize_stem(s) == "What is 2 + 2?"


def test_non_string_passthrough():
    assert normalize_stem(None) is None