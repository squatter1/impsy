from impsy import evaluation, heuristics
from impsy.evaluation import evaluate
from click.testing import CliRunner
from pathlib import Path
import numpy as np
import pytest


def test_model_params_from_filename():
    params = evaluation.model_params_from_filename(Path("eval_models/2-musicMDRNN-dim2-layers2-units64-mixtures5-scale12.tflite"))
    assert params == {"dim": 2, "layers": 2, "units": 64, "mixtures": 5}
    with pytest.raises(ValueError):
        evaluation.model_params_from_filename(Path("models/unnamed.tflite"))


def test_parse_log_file(tmp_path):
    log_file = tmp_path / "1-2d-mdrnn.log"
    log_file.write_text(
        "2025-05-09T01:23:12.000000,interface,0.5\n"
        "2025-05-09T01:23:12.250000,interface,0.6\n"
        "2025-05-09T01:23:13.250000,interface,0.7\n"
    )
    data = evaluation.parse_log_file(log_file)
    assert data.shape == (3, 2)
    assert np.allclose(data[:, 0], [0.25, 1.0, 1.0])  # last duration repeats
    assert np.allclose(data[:, 1], [0.5, 0.6, 0.7])
    (tmp_path / "empty.log").write_text("")
    assert evaluation.parse_log_file(tmp_path / "empty.log").shape == (0, 2)


def test_prediction_matches():
    actual = np.array([0.5, 60 / 127])
    assert evaluation.prediction_matches(np.array([0.52, 60 / 127]), actual, True, True)
    assert not evaluation.prediction_matches(np.array([0.6, 60 / 127]), actual, True, True)  # duration off by 20%
    assert evaluation.prediction_matches(np.array([0.6, 60 / 127]), actual, False, True)
    assert not evaluation.prediction_matches(np.array([0.5, 61 / 127]), actual, True, True)  # wrong note
    assert evaluation.prediction_matches(np.array([0.5, 61 / 127]), actual, True, False)


def test_pair_models_with_logs(tmp_path):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir()
    logs.mkdir()
    for i in (1, 2):
        (models / f"{i}-musicMDRNN-dim2-layers2-units64-mixtures5-scale12.tflite").touch()
        (logs / f"{i}-2d-mdrnn.log").touch()
    (logs / "3-2d-mdrnn.log").touch()
    pairs = evaluation.pair_models_with_logs(models, logs, dimension=2)
    assert [label for _, _, label in pairs] == ["model 1", "model 2"]
    # a single model is evaluated against every log
    (models / "2-musicMDRNN-dim2-layers2-units64-mixtures5-scale12.tflite").unlink()
    pairs = evaluation.pair_models_with_logs(models, logs, dimension=2)
    assert [label for _, _, label in pairs] == ["log 1", "log 2", "log 3"]


def test_heuristic_presets_build():
    for preset in heuristics.HEURISTIC_PRESETS:
        built = heuristics.build_heuristics(preset)
        assert len(built) == len(heuristics.HEURISTIC_NAMES)
        for memory_fn, heuristic_fn, weight in built:
            assert callable(memory_fn) and callable(heuristic_fn) and weight > 0
    assert len(heuristics.build_heuristics("improv", ["key_and_modal"])) == 1
    with pytest.raises(ValueError):
        heuristics.build_heuristic("nope", 1.0)


def test_heuristics_run_on_memory():
    """Each preset heuristic accepts a (duration, pitch) memory and branch without error."""
    rng = np.random.default_rng(0)
    memory = np.array([[0.25, (60 + rng.choice([0, 2, 4, 5, 7, 9, 11])) / 127] for _ in range(45)])
    branch = memory[-3:]
    for memory_fn, heuristic_fn, weight in heuristics.build_heuristics("improv"):
        value = heuristic_fn(memory_fn(memory), branch, weight)
        assert np.isfinite(value)


def test_evaluate_command_help():
    result = CliRunner().invoke(evaluate, ["--help"])
    assert result.exit_code == 0
    assert "--preset" in result.output


def test_evaluate_command_no_models(tmp_path):
    result = CliRunner().invoke(evaluate, ["--models-dir", str(tmp_path), "--logs-dir", str(tmp_path)])
    assert result.exit_code != 0
