from impsy import interaction
from impsy import utils
from impsy import heuristics
import pytest
from pathlib import Path
import numpy as np
import logging


@pytest.fixture(scope="session")
def default_config():
    """get the default config file."""
    config_path = Path("configs") / "default.toml"
    config = utils.get_config_data(config_path)
    return(config)


@pytest.fixture(scope="session")
def user_only_untrained_config():
    """get a config file without a neural network and in user-only mode."""
    config_path = Path("configs") / "user-only-example.toml"
    config = utils.get_config_data(config_path)
    return(config)


@pytest.fixture(scope="session")
def default_dimension(default_config):
    return default_config["model"]["dimension"]


@pytest.fixture(scope="session")
def logger(default_dimension, log_location):
    logger = interaction.setup_logging(default_dimension, location=log_location)
    return logger


def test_logging(logger, dimension):
    """Just sets up logging"""
    assert isinstance(logger, logging.Logger)
    values = np.random.rand(dimension - 1)
    interaction.log_interaction("tests", values, logger)
    interaction.close_log(logger)


@pytest.fixture(scope="session")
def default_neural_network(default_config):
    net = interaction.build_network(default_config)
    return net


def test_build_network(default_neural_network):
    pass


@pytest.fixture(scope="session")
def interaction_server(default_config, log_location):
    interaction_server = interaction.InteractionServer(default_config, log_location=log_location)
    return interaction_server

# @pytest.fixture(scope="session")
def untrained_interaction_server(user_only_untrained_config, log_location):
    # interaction_server = 
    interaction.InteractionServer(user_only_untrained_config, log_location=log_location)
    # return interaction_server

# def test_broken_interaction_server():
#     interaction_server = interaction.InteractionServer({})

def test_monitor_user_action(interaction_server):
    """Just tests creation of an interaction server object"""
    interaction_server.monitor_user_action()


def test_make_prediction(interaction_server, default_neural_network):
    interaction_server.make_prediction(default_neural_network)


def test_input_list(interaction_server):
    interaction_server.construct_input_list(0,0.0)


def test_dense_callback(interaction_server, default_dimension):
    values = np.random.rand(default_dimension - 1)
    interaction_server.dense_callback(values)


def test_send_values(interaction_server, default_dimension):
    values = np.random.rand(default_dimension - 1)
    interaction_server.send_back_values(values)


## prediction tree configuration


def test_prediction_tree_config_defaults():
    """No [prediction_tree] block means the tree is off with every default filled in."""
    settings = interaction.prediction_tree_config({"model": {"dimension": 2}})
    assert settings["enabled"] is False
    assert settings["preset"] == "improv"
    assert settings["heuristics"] == heuristics.HEURISTIC_NAMES
    assert settings["weights"] == heuristics.HEURISTIC_PRESETS["improv"]
    assert settings["memory_length"] == 45
    assert settings["time_limit_ms"] == 100.0


def test_prediction_tree_config_overrides():
    config = {
        "model": {"dimension": 2},
        "prediction_tree": {
            "enabled": True,
            "preset": "nottingham",
            "heuristics": ["key_and_modal", "interval_markov"],
            "time_limit_ms": 50,
            "weights": {"key_and_modal": 0.5},
        },
    }
    settings = interaction.prediction_tree_config(config)
    assert settings["enabled"] is True
    assert settings["heuristics"] == ["key_and_modal", "interval_markov"]
    assert settings["time_limit_ms"] == 50
    assert settings["weights"]["key_and_modal"] == 0.5
    assert settings["weights"]["tempo_and_swing"] == heuristics.HEURISTIC_PRESETS["nottingham"]["tempo_and_swing"]
    built = heuristics.build_heuristics(settings["preset"], settings["heuristics"], settings["weights"])
    assert [weight for _, _, weight in built] == [0.5, 0.25]


def test_prediction_tree_config_requires_dimension_two():
    config = {"model": {"dimension": 9}, "prediction_tree": {"enabled": True}}
    assert interaction.prediction_tree_config(config)["enabled"] is False


def test_prediction_tree_config_rejects_unknown_names():
    with pytest.raises(ValueError):
        interaction.prediction_tree_config({"model": {"dimension": 2}, "prediction_tree": {"preset": "jazz"}})
    with pytest.raises(ValueError):
        interaction.prediction_tree_config({"model": {"dimension": 2}, "prediction_tree": {"heuristics": ["nope"]}})
    with pytest.raises(ValueError):
        interaction.prediction_tree_config({"model": {"dimension": 2}, "prediction_tree": {"weights": {"nope": 1.0}}})


def test_default_config_file_has_prediction_tree(default_config):
    settings = interaction.prediction_tree_config(default_config)
    assert settings["enabled"] is False  # default.toml is 9D, the tree only applies to 2D
    assert settings["heuristics"] == heuristics.HEURISTIC_NAMES
