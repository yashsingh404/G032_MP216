# Model Training and MLflow Experiments

## What we predicted
Our target, `handle_min`, is the number of minutes needed to handle a helpline call. Because the target is numeric, this is a regression task.

## Original model
We used a scikit-learn pipeline with median numerical imputation (`queue_wait_sec`, `agent_exp_months`, `previous_calls`, `issue_count`, `hour`, `caller_age`), most-frequent categorical imputation + `OneHotEncoder(handle_unknown="ignore")` (`language`, `team`), and Random Forest Regression. Initial experiments were logged under the `MP216_Helpline_Regression` MLflow experiment. The original champion used 200 trees and had approximately 1.1583 minutes MAE on its initial chronological validation.

## Why we monitored and retrained
On the newer `later.csv` data, the original model's overall MAE degraded to about 3.5753 minutes. Drift monitoring in `findings.yaml` showed strong drift on `previous_calls` (PSI `6.582194`), while Team Beta had the highest group-level MAE at `6.389` minutes (versus `2.671` for Team Alpha and `2.971` for Team Gamma).

## Assignment-compliant comparison
In `src/mp216/assignment_v2.py`, `train.csv` (`fix_alpha_units=True`) and `later.csv` (`fix_alpha_units=False`) are cleaned and the 300 `later` records are chronologically sorted (`kind="stable"`). Using a 70/30 split (`int(len(later) * 0.7)`), the first 210 records were combined with the original 1,000 training records (1,210 total training rows) to train a candidate Random Forest (`n_estimators=300`, `random_state=42`). We evaluated both models on the remaining 90 check records under the `MP216_Assignment_V2` MLflow experiment:

| Model | Check-set MAE (minutes) |
|---|---:|
| Version 1 | 3.878471 |
| Retrained candidate | 1.913324 |

The candidate improved MAE by 50.67% (`(mae_v1 - mae_v2) / mae_v1 * 100`). The client's rule required at least a 12% reduction (`improvement >= 12`), so the candidate qualified for promotion (`decision = "promote"`).

## Model registry
Registered name: `MP216_Helpline_Model`.
* Version 1: original champion.
* Version 2: earlier development candidate.
* Version 3: assignment-compliant retrained model, now `champion` (`champion_version: 3` in `findings.yaml`, promoted separately after `assignment_v2.py` verified the threshold without auto-mutating the alias).

## Important limitation
The 90-row check set follows the assignment's 70/30 chronological split, but the first and last partitions share the date `2026-08-13` at the boundary. The results do not guarantee accuracy on all future data. September labels were not provided, so the 250 holdout predictions have no measurable MAE.