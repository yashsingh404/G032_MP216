# SCSE3040 MLOps --- Mini Project Student Handbook

Read this with your group on the first day. Keep it open while you work.
Your own problem is in `PROBLEM.md`; your group and members are in `GROUP.md`.

## 1. What this is, in one paragraph

Your group of four gets one small business problem from a made-up Indian
client, with its own data. You build the **whole MLOps pipeline** for it, the
same one you built step by step in the labs: clean data under version
control, tracked experiments, a registered model, a prediction API, a
container, automatic tests on GitHub, Kubernetes files, structured logs, drift
monitoring and a retrain. You submit **one public GitHub link**. A marking
tool reads your repository; then your faculty asks each of you questions.

It is the course **Assignment: 10 marks**.

**Key dates** (your problem e-mail has the full timeline):

| When | What |
|---|---|
| **Sun 29 Nov 2026, 11:59 PM** | **Deadline.** Every member has filled the form. |
| Mon 30 Nov -- Fri 4 Dec, 11:59 PM | **Grace period.** Accepted, with Timely 2 instead of 3. |
| After Fri 4 Dec | Very late: Timely 1. A group that never submits gets **0**. |
| Week starting Mon 30 Nov | **Viva** in your lab slot, each member separately. |

## 2. How you are marked (BTLAP Assignment rubric)

| Part | Marks | Who decides | How |
|---|---|---|---|
| Timely submission | 3 | the tool | From the time **you** filled the form (BTLAP rubric). On or before 29 Nov: 3. In the grace period (30 Nov -- 4 Dec): 2. Very late, incomplete (tool score below 40%), or not as per these rules (e.g. your link differs from your group's, or you did not fill the form while your group did): 1. Nobody in the group filled it: 0. |
| Completeness | 4 | the tool | The tool's score of your group's repository (section 11) turned into the rubric bands: 85% or more: 4; 65%: 3; 40%: 2; 15%: 1; below: 0. |
| Oral assessment | 3 | your faculty | A short viva **for each member separately**: 3, 2, 1.5, 1 or 0. |
| Individual contribution | (caps the oral) | the tool suggests, your faculty decides | High, Medium, Low or None, from your commits, whether the parts you claim in the form really exist in the repository, and the viva. Low keeps at most 1 oral mark; None keeps 0. |

**If nobody in your group fills the form, every member gets 0** (rubric
rule). Every mark comes with a written reason, which your faculty can show
you.

## 3. Rules

1. **One public repository per group.** Every member commits from **their own
   GitHub account**. The tool counts commits per person, and your faculty
   sees it.
2. **Use only your own problem's data.** The tool knows the exact fingerprint
   (hash) of your files. Results computed on another group's data score zero
   and are flagged.
3. **Do not copy code from another group.** The tool compares every file
   across all 263 repositories and lists identical files to the faculty.
4. **AI assistants (ChatGPT, Gemini, Claude, Copilot)** may be used like a
   tutor: to explain an error or an idea. They cannot do this project for
   you. Most marks come from **numbers that only exist after you run code on
   your own data** (your data's hash, your MLflow run ids, your holdout
   score, your CI run), and in the viva you will be asked to explain and
   change your own code live. Code you cannot explain earns an oral mark of 0.
5. **Never open `train.csv` in Excel and save it.** Excel changes the file,
   its hash stops matching, and the tool marks your data as "not yours". If
   that happens, extract the original file from your zip again.

## 4. Suggested roles (everyone still commits and everyone must explain all of it)

| Role | Owns | Labs that taught it |
|---|---|---|
| Data lead | DVC, the seven data checks, cleaning code, `findings.yaml` data part | P01--P05, L06 |
| Model lead | training, MLflow tracking and registry, holdout prediction | P02, P03, P06 |
| Service lead | FastAPI app, request rules, tests, Dockerfile | P05, P07, P08 |
| Ops lead | GitHub Actions, Kubernetes files, logging, drift and retraining | P09--P14 |

## 5. The repository layout (the tool looks in these places)

```
README.md                    group id, problem id, every member's enrolment no.
requirements.txt             every library pinned:  pandas==2.2.3
params.yaml                  your settings (paths, model settings); read with yaml.safe_load
src/<package>/__init__.py    your code, split into 2 or more modules
src/<package>/*.py
tests/test_*.py              3 or more pytest tests; at least one uses TestClient
data/raw/train.csv.dvc       DVC files (the CSVs themselves are NOT committed)
data/raw/later.csv.dvc
data/raw/holdout_features.csv
.dvc/config                  from `dvc init`
mlflow.db                    your MLflow database --- commit it
app/main.py                  FastAPI service: POST /predict and GET /health
models/model.joblib          the champion model the service loads
Dockerfile
.dockerignore
.github/workflows/ci.yml     runs pytest on every push
k8s/deployment.yaml
k8s/service.yaml
reports/findings.yaml        the form from your zip, filled in
predictions/holdout_predictions.csv
```

## 6. A plan in four stages

The time is short, so work on the stages in parallel where you can. Your
problem e-mail gives a date for each stage.

| Stage | Do this | Done when |
|---|---|---|
| 1 | Make the repo, add all four members, write the README. `dvc init`; `dvc add data/raw/train.csv data/raw/later.csv`; commit. Explore `train.csv` and hunt the seven data problems (section 7). | The `.dvc` files are on GitHub and the data part of `findings.yaml` is filled. |
| 2 | Write the cleaning code in `src/`. Train v1 with MLflow tracking (section 8). Try 3+ settings. Register the best model and give it the `champion` alias. Write tests. | `mlflow.db` on GitHub with 3+ runs and a registered model. |
| 3 | FastAPI app with the client's two rules (`PROBLEM.md` section 5). Dockerfile. GitHub Actions running pytest --- get it **green**. Kubernetes files with **your** names and replica count. JSON logging. | A green tick on the Actions tab. |
| 4 | `later.csv`: PSI per column, MAE per group, retrain v2, the promotion rule (`PROBLEM.md` section 7). Predict the holdout file. Finish `findings.yaml`. Check section 11. **Submit.** | The form is submitted before the deadline. |

## 7. The seven kinds of data problem

Some of these are hidden in your `train.csv` and some are not. For each,
`findings.yaml` asks: present, `true` or `false`, and if present, the details.
Write the code that finds each one in `src/` --- in the viva you will run it.

| Kind | What it means | How a careful person spots it |
|---|---|---|
| `leaking_column` | A column that is only filled in **after** the event, so it secretly contains the answer. A model that uses it looks perfect and fails in real life. | It is missing from `holdout_features.csv`. It agrees almost exactly with the target. |
| `unit_mixup` | One office (one value of the group column) recorded one column in **another unit**, for example metres instead of km. | Compare the column's minimum, average and maximum **for each group value**. One group is about 10, 12, 60, 100 or 1000 times bigger or smaller. `factor` is the **exact unit conversion** (1000 for km to metres, 12 for years to months), not a rough ratio. |
| `sentinel_value` | Someone typed a special number such as `-1`, `-999`, `999` or `9999` instead of leaving a reading blank. | `value_counts()` shows one impossible number repeated many times. |
| `duplicate_rows` | The same record appears twice. | `df.duplicated().sum()` gives the number of **extra** copies. |
| `stuck_sensor` | A device kept sending the same reading for a few weeks. | Sort by date. One column shows the same value on every row for a long run of days. Give the first and last date of the run. |
| `target_typos` | A few targets were typed with an extra zero: 10 times too big. | Sort the target. A small group of values sits far above all the others, with a big empty gap below them. |
| `spelling_variants` | One text column has the same category written in different ways (`Rain`, `rain `, `RAI.`). | `value_counts()` on each text column. `later.csv` has the clean spellings. `true_levels` is the number of real categories. |

Counts: count rows in `train.csv` **as given**. Counting after removing
duplicates is also accepted.

## 8. MLflow, exactly

- Track to a SQLite file in the repository root (MLflow 3 refuses `./mlruns`):
  `mlflow.set_tracking_uri("sqlite:///mlflow.db")`. Commit `mlflow.db`.
  You may add the model-file folder (`mlruns/` or `mlartifacts/`) to
  `.gitignore`; the tool reads only `mlflow.db`.
- In **every** run, log a parameter `data_sha256`: the SHA-256 of the
  original `train.csv` from your zip.
  `hashlib.sha256(open("data/raw/train.csv", "rb").read()).hexdigest()`
- Log a metric whose name contains `mae` (for example `mae_later`).
- Register your model. Version 1 is v1. Your retrained model is version 2.
  The model you choose gets the alias `champion`
  (`MlflowClient().set_registered_model_alias(name, "champion", version)`).
- If `log_model` fails with "untrusted types", add
  `serialization_format="cloudpickle"` to the call. This is MLflow 3.15's
  safer default; it is not your bug.

## 9. The service, the container, CI and Kubernetes

- **API.** `POST /predict` takes exactly the fields listed in your
  `PROBLEM.md` and returns the prediction. `GET /health` returns
  `{"status": "ok"}`. The client's two rules go **in the pydantic schema**
  with `Field(..., le=...)` / `Field(..., ge=...)`, so bad input gets 422.
- **Logging.** Use the `logging` module with a JSON formatter (P10): one
  JSON line per prediction, written to the screen (stdout), never to a file.
  Never log a customer's personal details.
- **Dockerfile.** `python:3.12-slim`; copy `requirements.txt` and
  `pip install` **before** copying your code; `EXPOSE 8000`; start uvicorn
  with `--host 0.0.0.0 --port 8000`. Add a `.dockerignore`.
- **CI.** `.github/workflows/ci.yml` must start with the key `on:` (not
  `true:`, which you get if you save the file through `yaml.safe_dump`),
  run on `push`, install `requirements.txt` and run `pytest`. The tool checks
  for a **green run on GitHub** before your submission time. Your tests must
  not need the CSVs (they are not in git): build small test data inside
  the test.
- **Kubernetes.** Use the exact Deployment name, Service name, image and
  replica count from `PROBLEM.md` section 6. The Service selector must match
  the Pod labels. Readiness and liveness probes both on `/health`; liveness
  waits longer. No cluster is needed: the files are what is marked.

## 10. Monitoring, retraining and the final prediction

Follow `PROBLEM.md` sections 7 and 8 exactly: PSI for every numeric column
(train vs later), v1's MAE per group, retrain on train plus the first 70% of
`later.csv` (sorted by date), compare on the last 30%, apply the client's
promotion rule, and predict `holdout_features.csv` with your champion.
`predictions/holdout_predictions.csv` has exactly the columns
`record_id,prediction` and one row per holdout record.

## 11. What the tool checks (out of 100)

| Section | Points |
|---|---|
| A. README, pinned requirements, params.yaml read by code, `src/` package | 10 |
| B. 3+ tests, and a TestClient test | 6 |
| C. DVC set up; `train.csv.dvc` has your original file's hash | 6 |
| D. The seven data problems: present or not, and the details | 14 |
| E. MLflow: 3+ runs, `data_sha256`, an MAE metric, your run id, `@champion`, 2+ versions | 12 |
| F. `/predict`, `/health`, every field in the schema, the client's two rules | 10 |
| G. Dockerfile done right, `.dockerignore` | 8 |
| H. CI: triggers on push, runs pytest, a green run on GitHub | 8 |
| I. Kubernetes: names, image, replicas, matching selectors, probes, ports | 8 |
| J. JSON logging | 4 |
| K. Most drifted column, worst-served group, a decision that follows the rule | 8 |
| L. Holdout predictions: every row, and an MAE close to a good model's | 6 |

## 12. Submitting

1. Make the repository **public**. Open the link in a private browser
   window to check.
2. **Every member fills the submission form separately** (the link is in
   your problem e-mail), signed in with their Bennett account. It asks for
   your group and problem id, your GitHub username, your group's repository
   link (**the same link for everyone**), the parts you personally built, a
   short description of your work, the files you wrote, and your hours. Be
   honest: the tool checks your claims against the repository and your
   commits, and the viva checks them against you.
3. The tool marks your group's **last commit made before the last member's
   form time** (but not later than the grace period). Work pushed later is
   ignored. You may fill the form again; your **last** response counts, and
   its time decides your Timely marks --- so filling it again after 29 Nov
   makes it a grace-period submission (Timely 2).

## 13. Before you submit: a self-check

- [ ] Public link opens in a private window.
- [ ] README has the group id, the problem id and all enrolment numbers.
- [ ] `findings.yaml` has no `null` left except where a problem is absent.
- [ ] `mlflow.db` is committed and the alias `champion` is set.
- [ ] The Actions tab shows a green tick.
- [ ] `predictions/holdout_predictions.csv` has every `record_id`.
- [ ] Every member has commits under their own name.
- [ ] **Every member has filled the form**, with the same repository link.
- [ ] Every member can explain every file.

## 14. Questions students always ask

- **"Our MAE is not the best possible."** It does not need to be. A sensible
  model passes the bar; the marks are for the pipeline.
- **"Can we use a different model?"** Yes. Any scikit-learn regressor.
- **"A member is not working."** Tell your faculty early. The contribution
  column exists for this, and commits are counted per person.
- **"We lost our zip."** Ask your faculty; problems can be re-issued exactly.
