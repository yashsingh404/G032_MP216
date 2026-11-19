# Helpline Call Handling Time Prediction (G032_MP216)

## About the Project

This project is part of our MLOps coursework. The aim is to build a machine learning system that predicts how long a helpline call will take to handle.

Instead of only training a model and checking its accuracy, we wanted to work through the complete machine learning process. This includes cleaning the data, tracking experiments, building an API, running the application in Docker, monitoring changes in the data, and improving the model when its predictions become less reliable.

We used a Random Forest Regressor for the predictions and MLflow to keep track of our experiments and model versions.

**Group:** G032  
**Problem:** MP216  
**GitHub Repository:** https://github.com/yashsingh404/G032_MP216

## Tools and Technologies

We used the following tools throughout the project:

| Tool | How we used it |
|---|---|
| Python | Main programming language |
| Pandas and NumPy | Data analysis and preprocessing |
| Scikit-learn | Training the regression models |
| MLflow | Tracking experiments and managing models |
| DVC | Version control for datasets |
| FastAPI | Creating the prediction API |
| Docker | Running the API in a container |
| GitHub Actions | Automatically running tests |
| Pytest | Testing our code |

## Understanding the Data

We were given three datasets:

| Dataset | Records | Usage |
|---|---:|---|
| `train.csv` | 1,000 | Training the original model |
| `later.csv` | 300 | Monitoring drift and improving the model |
| `holdout_features.csv` | 250 | Generating final predictions |

Our target column is `handle_min`, which represents the number of minutes needed to handle a call.

Some of the useful features include the waiting time, agent experience, number of previous calls, number of issues, caller age, language, and team.

### Data Cleaning

Before training the model, we checked the data for problems.

We found a few issues that needed attention:

- Some waiting-time values for Team Alpha in the original training data were recorded using a different unit.
- The `caller_age` column contained `-999`, which was being used instead of a missing value.
- Some language names were written in different ways.
- The `hour` column contained suspicious repeated values between February 2 and February 21, 2026.

We handled these issues in our cleaning code instead of changing the original CSV files.

One important detail was that the later and holdout datasets did not need the same Team Alpha unit correction as the original training data. Applying that correction everywhere would have introduced new errors.

## Training Our First Model

We started with a Random Forest Regressor.

For preprocessing, we used missing-value imputation for numerical and categorical features, along with one-hot encoding for categorical columns.

We tried different Random Forest configurations and tracked the results using MLflow.

Our first selected model used 200 trees and achieved a validation MAE of approximately **1.1583 minutes**.

We registered this model in MLflow under the name:

`MP216_Helpline_Model`

It became Version 1 and was initially assigned the `champion` alias.

## Building the Prediction API

After training the model, we built a FastAPI application so predictions could be requested through an HTTP API.

The application has two endpoints:

- `GET /health` checks whether the API is running.
- `POST /predict` accepts call details and returns a predicted handling time.

To start the API locally:

```bash
python -m uvicorn app.main:app --reload
```

The interactive API documentation can be opened at:

http://127.0.0.1:8000/docs

We tested both the health endpoint and the prediction endpoint using pytest.

By default, the API loads the model with the MLflow `champion` alias.

## Running the API with Docker

We also created a Dockerfile so the application could run in a container.

To build the image:

```bash
docker build -t mp216-api .
```

To run it:

```bash
docker run -d --name mp216-api -p 8000:8000 mp216-api
```

The API can then be accessed through:

http://localhost:8000

We tested the container locally and confirmed that it could return predictions.

The Docker image uses an exported copy of the model. This means that if we promote a new model in MLflow, the existing Docker image will still use its previously exported model until we rebuild it.

## Automated Testing and GitHub Actions

We added automated tests to make sure the API and monitoring functions were working as expected.

Our tests include:

- API health checks
- Prediction endpoint checks
- Data drift calculation tests
- MLflow model registration and loading checks

To run the API and monitoring tests:

```bash
python -m pytest tests/test_api.py tests/test_monitor.py -v
```

To run the local MLflow registry tests:

```bash
python -m pytest tests/test_model_registry.py -v
```

We successfully ran 7 API and monitoring tests and 3 MLflow registry tests.

We also set up GitHub Actions to run the portable tests whenever changes are pushed to GitHub. Our latest verified workflow run passed successfully.

The MLflow registry tests are kept separate from GitHub Actions because they rely on model files and a database available on our local machine.

## Monitoring Changes in the Data

One of the main parts of this project was checking whether our original model would still work well on newer calls.

When we received `later.csv`, we compared it with the original training data.

We calculated the Population Stability Index (PSI) for six numerical features using 10 equal-width bins.

| Feature | PSI |
|---|---:|
| `queue_wait_sec` | 0.034098 |
| `agent_exp_months` | 0.025137 |
| `previous_calls` | **6.582194** |
| `issue_count` | 0.011309 |
| `hour` | 0.047464 |
| `caller_age` | 0.053933 |

The biggest change was in `previous_calls`.

This was also noticeable when we compared the average number of previous calls between the training and later datasets. The average had increased considerably.

### How Well Did Our Original Model Perform?

We evaluated our original champion on the later dataset.

The overall MAE increased to approximately **3.5753 minutes**, which was much higher than the error on its original validation set.

We also checked the MAE for each team.

| Team | MAE (minutes) |
|---|---:|
| Team Alpha | 2.671 |
| Team Beta | **6.389** |
| Team Gamma | 2.971 |

Team Beta had the highest error.

We also noticed that calls involving a large number of previous calls were more difficult for the original model to predict accurately.

These findings gave us a clear reason to try retraining.

## Improving the Model

For our first round of improvement, we experimented with additional training data and compared several Random Forest configurations.

The best earlier candidate used 300 trees and achieved an August validation MAE of approximately **1.7966 minutes**. We registered it as Version 2 for development.

We then followed the assignment's required retraining procedure.

We sorted `later.csv` by date and divided it into two parts:

- First 70% (210 records) for retraining
- Last 30% (90 records) for checking model performance

We combined those 210 records with the original 1,000 training records.

This gave us **1,210 training records** for the updated model.

We trained another Random Forest with 300 trees and evaluated both the original model and the new candidate on the same 90 check records.

### Final Model Comparison

| Model | MAE (minutes) |
|---|---:|
| Original model (Version 1) | 3.878471 |
| Retrained model | **1.913324** |

The updated model reduced MAE by **50.67%** on this check set.

Our assignment required at least a 12% reduction in MAE before promoting a new model. Since the improvement was above that threshold, we decided to promote the updated model.

We registered it as **Version 3** and changed the MLflow `champion` alias to Version 3.

The earlier model versions are still available in MLflow.

This improvement was measured on the assignment's check set, so it should not be treated as a guarantee that the model will perform equally well on every future dataset. The 70/30 split also placed some August 13 records on both sides of the boundary because multiple records shared that date.

## Making the Final Predictions

Once Version 3 became the champion, we used it to generate predictions for the September holdout dataset.

The output was saved to:

`predictions/holdout_predictions.csv`

The file contains the two required columns:

```csv
record_id,prediction
R50001,16.841633
R50002,18.569033
R50003,11.863867
R50004,19.227767
R50005,21.238433
```

We verified that:

- All 250 holdout records received a prediction.
- There were no missing predictions.
- All predictions were finite and nonnegative.
- The original record IDs were preserved.

Since the holdout file does not contain actual call handling times, we could not calculate MAE for these final predictions.

## Important Project Files

Here are some of the main files we worked with:

| File | Purpose |
|---|---|
| `src/mp216/data_cleaning.py` | Handles data cleaning |
| `src/mp216/monitor.py` | Checks changes in the data |
| `src/mp216/performance_monitor.py` | Measures model performance |
| `src/mp216/error_analysis.py` | Analyzes prediction errors |
| `src/mp216/psi_monitor.py` | Calculates PSI values |
| `src/mp216/candidate_train.py` | Trains development candidates |
| `src/mp216/assignment_v2.py` | Performs the required retraining and evaluation |
| `src/mp216/final_predictions.py` | Generates final holdout predictions |
| `app/main.py` | FastAPI application |
| `tests/` | Automated tests |
| `.github/workflows/ci.yml` | GitHub Actions workflow |
| `findings.yaml` | Records the assignment findings |

## Reproducing the Project

The Python dependencies are listed in `requirements.txt`.

To install them:

```bash
pip install -r requirements.txt
```

The project also uses DVC to track datasets and MLflow to manage model experiments.

Some data and model artifacts are stored locally rather than directly in GitHub. Reproducing all experiments on another machine therefore requires access to the corresponding DVC data and MLflow artifacts.

## What We Learned

The most useful lesson from this project was that getting a good score while training a model is not the end of the work.

Our first model performed well on its initial validation data, but its error increased when we tested it on newer calls. Monitoring helped us identify the change in `previous_calls` and the particularly high errors for Team Beta.

Retraining on more recent examples improved the model significantly on the required check set.

We also learned how the different MLOps tools fit together. MLflow helped us keep track of models, FastAPI made our predictions accessible, Docker gave us a consistent way to run the API, and GitHub Actions helped us catch problems when the code changed.

## Conclusion

We built and tested a helpline call handling time prediction pipeline covering data cleaning, model training, experiment tracking, API serving, containerization, monitoring, retraining, and automated testing.

Our final retrained Random Forest achieved a check-set MAE of **1.913324 minutes**, compared with **3.878471 minutes** for the original model. This was a **50.67% improvement**, so the new model was promoted to Version 3 in MLflow.

We also generated the required 250 predictions for the September holdout dataset.

Overall, this project gave us practical experience with how a machine learning model can be developed, monitored, improved, and managed over time.

---

**Group:** G032  
**Problem ID:** MP216  
**Final Champion:** MLflow Model Version 3  
**Final Check-Set MAE:** 1.913324 minutes  
**GitHub:** https://github.com/yashsingh404/G032_MP216

## Team Members

| Name | Enrolment Number |
|---|---|
| Yash Singh | S24CSEU2682 |