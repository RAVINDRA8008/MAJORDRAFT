"""Tests for label mapper."""

import pytest

from src.data.label_mapper import LabelMapper


@pytest.fixture
def mapper():
    return LabelMapper()


def test_deap_happy(mapper):
    assert mapper.deap_label(7.0, 7.0) == 0  # High V, High A → Happy


def test_deap_sad(mapper):
    assert mapper.deap_label(2.0, 2.0) == 1  # Low V, Low A → Sad


def test_deap_angry(mapper):
    assert mapper.deap_label(2.0, 7.0) == 2  # Low V, High A → Angry


def test_deap_neutral(mapper):
    assert mapper.deap_label(7.0, 2.0) == 3  # High V, Low A → Neutral/Relaxed


def test_iemocap_excited_maps_to_happy(mapper):
    assert mapper.iemocap_label("exc") == 0


def test_iemocap_unknown(mapper):
    assert mapper.iemocap_label("fru") == -1
