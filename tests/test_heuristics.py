from impsy import heuristics
import numpy as np
import pytest


C_MAJOR = np.array([0, 2, 4, 5, 7, 9, 11])


def notes_to_memory(notes, duration=0.5):
    """Builds a (duration, pitch) memory from MIDI note numbers."""
    return np.array([[duration, note / 127] for note in notes])


## key and mode


def test_root_conformity():
    major = heuristics.Scale(C_MAJOR, name="Major")
    assert major.root_conformity(np.array([60, 62, 64, 65, 67]), 0) == pytest.approx(1.0)
    # every note out of C major (root 0) lands in scale for root 1
    assert major.root_conformity(np.array([61, 63, 66]), 0) < 0
    assert major.root_conformity(np.array([61, 63, 66]), 1) > 0


def test_key_conformity_finds_c_major():
    conformity, scale, root = heuristics.key_conformity(np.array([60, 62, 64, 65, 67, 69, 71, 72]))
    assert conformity == pytest.approx(1.0)
    assert scale.name == "Major"
    assert root == 0


def test_key_conformity_below_threshold_returns_none():
    chromatic = np.arange(60, 72)
    conformity, scale, root = heuristics.key_conformity(chromatic, min_key_conformity=0.75)
    assert conformity == 0 and scale is None and root is None


def test_mode_conformity():
    major = heuristics.Scale(C_MAJOR, name="Major")
    # C E G on the C major triad
    assert major.mode_conformity(np.array([0, 4, 7]), 0, 0) == pytest.approx(1.0)
    # D F A on mode 0 has no triad notes
    assert major.mode_conformity(np.array([2, 5, 9]), 0, 0) < 0
    # no notes in the scale
    assert major.mode_conformity(np.array([1, 3]), 0, 0) == 0


def test_mode_conformity_without_a_triad():
    # a mode with no buildable triad is not evaluated
    weird = heuristics.Scale(np.array([0, 1, 6]), name="weird")
    assert weird.mode_conformity(np.array([0, 1, 6]), 0, 0) == -1
    # a three note scale is its own triad
    whole_tone = heuristics.Scale(np.array([0, 2, 4, 6, 8, 10]), name="whole tone")
    assert whole_tone.mode_conformity(np.array([0, 4, 8]), 0, 0) == pytest.approx(1.0)
    assert heuristics.Scale(np.array([0, 4, 7]), name="triad").mode_conformity(np.array([0, 4]), 0, 0) == pytest.approx(1.0)


def test_key_and_modal_heuristic_prefers_in_key_branch():
    memory = notes_to_memory([60, 62, 64, 65, 67, 69, 71, 72] * 4)
    memory_tuple = heuristics.key_and_modal_memory(memory, min_key_conformity=0.7)
    in_key = heuristics.key_and_modal_conformity_heuristic(memory_tuple, notes_to_memory([64, 67, 72]), 1.0)
    out_of_key = heuristics.key_and_modal_conformity_heuristic(memory_tuple, notes_to_memory([61, 63, 66]), 1.0)
    assert 0 <= in_key < out_of_key


def test_key_and_modal_heuristic_is_zero_without_a_key():
    memory_tuple = heuristics.key_and_modal_memory(notes_to_memory(range(60, 72)), min_key_conformity=0.7)
    assert heuristics.key_and_modal_conformity_heuristic(memory_tuple, notes_to_memory([60, 61]), 1.0) == 0


## tempo and swing


def test_estimate_tempo_and_swing():
    # quarter notes at 120 bpm
    tempo, swing, ratio, deviation = heuristics.estimate_tempo_and_swing(np.full(16, 0.5))
    assert tempo == 120
    assert swing == "none"
    assert deviation == pytest.approx(0.0)


def test_tempo_and_swing_heuristic():
    memory_tuple = heuristics.tempo_and_swing_memory(notes_to_memory(range(60, 76), duration=0.5))
    on_tempo = heuristics.tempo_and_swing_heuristic(memory_tuple, notes_to_memory([60, 62], duration=0.5), 1.0)
    off_tempo = heuristics.tempo_and_swing_heuristic(memory_tuple, notes_to_memory([60, 62], duration=0.43), 1.0)
    assert on_tempo == 0
    assert off_tempo > 0


def test_tempo_and_swing_heuristic_is_zero_without_a_tempo():
    rng = np.random.default_rng(0)
    memory = np.column_stack([rng.uniform(0.1, 1.0, 40), np.full(40, 60 / 127)])
    memory_tuple = heuristics.tempo_and_swing_memory(memory)
    assert memory_tuple[3] > 0.08
    assert heuristics.tempo_and_swing_heuristic(memory_tuple, memory[:3], 1.0, max_tempo_deviation=0.08) == 0


## markov heuristics


def test_interval_markov_prefers_seen_intervals():
    # a repeating rising scale, so the interval after +2 is usually +2
    memory = notes_to_memory([60, 62, 64, 66, 68, 70] * 5)
    model = heuristics.interval_markov_memory(memory, order=1)
    assert model[3] is True
    likely = heuristics.interval_markov_heuristic(model, notes_to_memory([62, 64, 66]), 1.0)
    unlikely = heuristics.interval_markov_heuristic(model, notes_to_memory([67, 61, 70]), 1.0)
    assert likely < unlikely


def test_interval_markov_invalid_with_short_memory():
    model = heuristics.interval_markov_memory(notes_to_memory([60, 62]), order=2)
    assert model[3] is False
    assert heuristics.interval_markov_heuristic(model, notes_to_memory([60, 61, 62]), 1.0) == 0


def test_time_multiple_markov_prefers_seen_rhythms():
    # alternating long and short notes
    durations = [0.5, 0.25] * 10
    memory = np.array([[d, 60 / 127] for d in durations])
    model = heuristics.time_multiple_markov_memory(memory, order=1)
    assert model[3] is True
    same_rhythm = np.array([[0.5, 0.5], [0.25, 0.5], [0.5, 0.5]])
    new_rhythm = np.array([[1.5, 0.5], [0.1, 0.5], [0.8, 0.5]])
    assert heuristics.time_multiple_markov_heuristic(model, same_rhythm, 1.0) < heuristics.time_multiple_markov_heuristic(model, new_rhythm, 1.0)


def test_repetition_markov_prefers_repeated_patterns():
    pattern = [60, 64, 67, 72]
    memory = notes_to_memory(pattern * 8)
    model = heuristics.repetition_markov_memory(memory, order=2)
    assert model[3] is True
    repeated = heuristics.repetition_markov_heuristic(model, notes_to_memory([60, 64, 67, 72]), 1.0)
    novel = heuristics.repetition_markov_heuristic(model, notes_to_memory([61, 70, 63, 66]), 1.0)
    assert repeated < novel


def test_heuristics_scale_with_multiplier():
    memory = notes_to_memory([60, 62, 64, 66, 68, 70] * 5)
    model = heuristics.interval_markov_memory(memory, order=1)
    branch = notes_to_memory([67, 61, 70])
    assert heuristics.interval_markov_heuristic(model, branch, 2.0) == pytest.approx(2 * heuristics.interval_markov_heuristic(model, branch, 1.0))
