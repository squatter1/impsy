from impsy.mcts_prediction_tree import MCTSNode, MCTSPredictionTree
from impsy import heuristics
import numpy as np
import pytest
import time


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def fake_model(rng):
    """Stands in for an MDRNN: predict returns dummy GMM params and states, sample draws a (duration, pitch)."""
    def predict(prev_output, init_lstm_states=None):
        return np.zeros(3), [np.zeros(1)]

    def sample(gmm_params):
        return np.array([0.25 * rng.choice([0.5, 1.0, 2.0]), (60 + rng.integers(0, 12)) / 127])

    return predict, sample


@pytest.fixture
def memory(rng):
    return np.array([[0.25, (60 + rng.choice([0, 2, 4, 5, 7, 9, 11])) / 127] for _ in range(45)])


def make_tree(fake_model, memory, heuristic_functions=None, **kwargs):
    predict, sample = fake_model
    return MCTSPredictionTree(
        root_output=memory[-1].copy(),
        initial_lstm_states=[np.zeros(1)],
        predict_function=predict,
        sample_function=sample,
        initial_memory=memory[:-1],
        heuristic_functions=heuristic_functions if heuristic_functions is not None else heuristics.build_heuristics("improv"),
        simulation_depth=2,
        greedy_weight=0.4,
        exploration_weight=0.1,
        **kwargs,
    )


## nodes


def test_node_snaps_pitch_to_semitones():
    # pitch is scaled by 1.27, rounded to 2 decimal places, then scaled back, so 0.4712 -> 0.6 / 1.27
    node = MCTSNode(np.array([0.3, 0.4712]), snap_dp=[None, 2])
    assert node.output[1] == pytest.approx(0.6 / 1.27)
    assert round(node.output[1] * 127) == 60
    assert node.output[0] == 0.3  # duration not snapped


def test_node_starts_unvisited():
    node = MCTSNode(np.array([0.3, 0.5]))
    assert node.visits == 0 and node.value == 0.0
    assert node.best_value == -np.inf
    assert node.children == []


def test_most_visited_child():
    parent = MCTSNode(np.array([0.3, 0.5]))
    for visits in (2, 5, 1):
        child = parent.add_child(np.array([0.3, 0.5]), snap_dp=[None, 2])
        child.visits = visits
    assert parent.most_visited_child().visits == 5


## search


def test_search_runs_requested_iterations(fake_model, memory):
    tree = make_tree(fake_model, memory)
    output, lstm_states = tree.search(memory=memory[:-1], max_iterations=50)
    assert tree.get_num_branches() == 50
    assert tree.get_num_nodes() > 50
    assert output.shape == (2,)
    assert any(np.array_equal(output, child.output) for child in tree.root.children)


def test_search_requires_a_limit(fake_model, memory):
    with pytest.raises(ValueError):
        make_tree(fake_model, memory).search(memory=memory[:-1])


def test_search_respects_time_limit(fake_model, memory):
    tree = make_tree(fake_model, memory)
    start = time.time()
    tree.search(memory=memory[:-1], time_limit_ms=50)
    assert time.time() - start < 1.0
    assert tree.get_num_branches() > 0


def test_search_returns_most_visited_child(fake_model, memory):
    tree = make_tree(fake_model, memory)
    output, _ = tree.search(memory=memory[:-1], max_iterations=100)
    assert np.array_equal(output, tree.root.most_visited_child().output)


def test_backpropagation_updates_values(fake_model, memory):
    """Results are negated penalties, so values are <= 0 and best_value must track the best of them."""
    tree = make_tree(fake_model, memory)
    tree.search(memory=memory[:-1], max_iterations=100)
    assert tree.root.visits == 100
    for child in tree.root.children:
        if child.visits:
            assert child.value <= 0
            assert -np.inf < child.best_value <= 0
            assert child.best_value >= child.value / child.visits


def test_children_are_distinct(fake_model, memory):
    tree = make_tree(fake_model, memory)
    tree.search(memory=memory[:-1], max_iterations=100)
    outputs = [tuple(np.round(child.output, 6)) for child in tree.root.children]
    assert len(outputs) == len(set(outputs))


def test_progressive_widening_limits_children(fake_model, memory):
    k, alpha = 2.5, 0.25
    tree = make_tree(fake_model, memory, progressive_widening_k=k, progressive_widening_alpha=alpha)
    tree.search(memory=memory[:-1], max_iterations=100)
    assert 1 <= len(tree.root.children) <= max(1, int(k * tree.root.visits ** alpha))


def test_heuristics_receive_full_branch(fake_model, memory):
    """Each heuristic sees the branch from root plus the simulated notes, as (n, 2) rows."""
    seen = []
    recording = [(lambda mem: None, lambda mem, branch, weight: seen.append(branch.shape) or 0.0, 1.0)]
    tree = make_tree(fake_model, memory, heuristic_functions=recording)
    tree.search(memory=memory[:-1], max_iterations=20)
    assert len(seen) == 20
    assert all(shape[1] == 2 and shape[0] >= 1 + 1 + 2 for shape in seen)  # root, selected node, two simulated


def test_set_root_reuses_subtree(fake_model, memory):
    tree = make_tree(fake_model, memory)
    output, _ = tree.search(memory=memory[:-1], max_iterations=50)
    chosen = next(child for child in tree.root.children if np.array_equal(child.output, output))
    old_root = tree.root
    tree.set_root(output)
    assert tree.root is chosen
    assert tree.root.parent is old_root
    assert old_root.children == []


def test_set_root_rejects_unknown_output(fake_model, memory):
    tree = make_tree(fake_model, memory)
    tree.search(memory=memory[:-1], max_iterations=10)
    with pytest.raises(ValueError):
        tree.set_root(np.array([9.0, 9.0]))


def test_best_branch_starts_at_root(fake_model, memory):
    tree = make_tree(fake_model, memory)
    tree.search(memory=memory[:-1], max_iterations=50)
    branch = tree.get_best_branch()
    assert branch.shape[1] == 2
    assert np.array_equal(branch[0], tree.root.output)
    assert len(branch) >= 2
