# Evaluation

Models, data and scripts used to evaluate MCTS-guided prediction against the plain MDRNN. See the `evaluate` command:

    poetry run ./start_impsy.py evaluate --help

## Contents

- `models/`: 15 2D MDRNNs (2 layers, 64 units, 5 mixtures, scale 12), one per improvisation log. Model `i` pairs with log `i`.
- `logs/`: 15 IMPSY logs of keyboard improvisations, one note per line as `timestamp,interface,pitch`.
- `datasets/`: the `.npz` training datasets used to train each model.
- `models_nottingham/`: one 128-unit model trained on melodies from the [Nottingham dataset](https://ifdo.ca/~seymour/nottingham/nottingham.html). The logs are not included, generate them with `scripts/convert_midi_to_logs.py` and select a subset with `scripts/select_eval_logs.py`.
- `models_shortened/`: five models trained on shortened logs, kept for reference. The shortened logs are not included.
- `playbacks/`: terminal transcripts of the qualitative sessions comparing MCTS and MDRNN in free and structured improvisation.
- `scripts/`: figure generation and data preparation, see below.

## Reproducing the results

    poetry run ./start_impsy.py evaluate                              # all heuristics, improv preset
    poetry run ./start_impsy.py evaluate -H key_and_modal             # one heuristic
    poetry run ./start_impsy.py evaluate --match pitch                # pitch accuracy only
    poetry run ./start_impsy.py evaluate -M evaluation/models_nottingham -L <nottingham logs> -P nottingham

## Scripts

Figures are written to `evaluation/figures/`. The plotting scripts need `scipy` and `matplotlib`.

- `results.py`: the accuracy and timing numbers behind the figures.
- `plot_column.py`: per-heuristic accuracy bars, e.g. `--corpus nottingham --metric pitch`.
- `plot_accuracy_vs_iterations.py`: accuracy against MCTS iterations with an asymptotic fit, `--bootstrap` adds confidence intervals.
- `plot_search_time.py`: search time against MCTS iterations.
- `convert_midi_to_logs.py`: converts a folder of MIDI melodies to IMPSY logs.
- `select_eval_logs.py`: randomly selects converted logs until a note count is reached.
- `rename_logs.py`: renames `i-n.log` files to the `i-n-2d-mdrnn.log` form the dataset command expects.
