# Experiment Outputs

This folder stores isolated experiment outputs so baseline project artifacts
remain untouched.

Expected per-run layout:

- `results/experiments/<run_name>/summary.json`
- `results/experiments/<run_name>/full_eval.json`
- `checkpoints/experiments/<run_name>/<run_name>_best.pt`
- `checkpoints/experiments/<run_name>/<run_name>_history.json`

Run the starter script:

```bash
python scripts/train_multiclass_experiment.py --config configs/multiclass_exp_v1.yaml
```
