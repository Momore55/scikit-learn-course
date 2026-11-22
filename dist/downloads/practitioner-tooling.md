# Optional practitioner tooling

These are extensions to the scikit-learn course, not prerequisites for finishing it. The primary lessons and applied labs run in the recorded environment. Commands below are guided next steps for separately installed tools; they were not executed as part of that environment and have no fabricated output.

## 1. Experiment tracking with MLflow

Start with the Chapter 35 JSON experiment record: dataset identity, split, hyperparameters, environment, metric, artifact reference, and actual code revision. A tracker makes these searchable across runs; it does not make an inappropriate evaluation valid.

In a separate environment, install MLflow following its official quickstart. Inside the fitting script, the core pattern is:

```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("regularization_C", chosen_C)
    mlflow.log_metric("development_macro_f1", development_macro_f1)
    mlflow.log_artifact("experiment.json")
```

`chosen_C` and `development_macro_f1` must come from your own actual experiment. Also record the data reference, split definition, and code revision. A logged artifact may be readable by other users of a shared tracker: choose an appropriate storage location and access policy.

Official documentation: https://mlflow.org/docs/latest/tracking/

Exercise: log two runs from a fixed development split, find their differences, and reproduce one using its manifest. Keep final-test evaluation separate from choosing the winning run.

## 2. Data versioning with DVC

A SHA-256 hash identifies exact bytes but cannot recover missing bytes. DVC can track data references alongside Git while data is retained in a separately configured storage location. Access terms still determine what may be stored or shared.

After installing DVC in a separate project, an introductory sequence is:

```sh
dvc init
dvc add data/training.csv
git add data/training.csv.dvc data/.gitignore .dvc
git commit -m "Record permitted training data version"
```

This assumes you already have Git, a permitted local CSV, and an appropriate `.gitignore`. It does not configure a remote, back up the data, or authorize public redistribution. Configure a suitable remote and use DVC's documented push/pull workflow if durable recovery is needed. A private Git repository does not automatically make a separate data remote private.

Official start: https://doc.dvc.org/start

Exercise: track two allowed versions, switch between them, and record which model and evaluation were paired with each version. Preserve the evaluation partition or explicitly document why it changed.

## 3. Containerized reproducibility with Docker

Containers package user-space dependencies; they do not freeze every hardware, operating-system kernel, data, or external-service effect. Use the recorded Python and package versions, and pin an image digest when strict base-image identity matters.

A starting Dockerfile for serving the static course locally is provided as `Dockerfile.static` in the repository:

```sh
docker build -f Dockerfile.static -t scikit-studio .
docker run --rm -p 8000:8000 scikit-studio
```

For a Python prediction service, build a separate image with the model's requirements, trusted artifact and input schema. Do not copy secrets, local datasets, or arbitrary untrusted model files into images. Validate reloaded predictions inside the target environment before shipping.

Official guide: https://docs.docker.com/guides/python/

Exercise: reproduce a saved pipeline's fixed-record predictions inside your own container and report any differences with a suitable numerical tolerance.

## 4. Explanation beyond PDP: SHAP and LIME

SHAP decomposes a model output relative to a selected reference/background and explainer assumptions. LIME fits a local surrogate around a prediction using a perturbation neighborhood. Both depend on representation, background data, correlation, and which output scale is explained. They do not establish causal effects or prove that a decision is fair.

Start with Chapter 32's permutation, PDP and ICE labs. Then choose a separately installed explainer, record its version, specify the background and output scale, and explain one correctly and one incorrectly predicted example. Check whether changing the background or neighborhood changes the explanation.

SHAP official documentation: https://shap.readthedocs.io/en/stable/
LIME project documentation: https://github.com/marcotcr/lime

Exercise: compare a global permutation result with a local explanation for one case. Explain why disagreement may reflect different questions rather than an implementation error.

## 5. Fairness and drift are ongoing investigations

The Chapter 33 subgroup lab supplies counts and error rates; Chapter 47 distinguishes schema failures, missingness, input drift, and outcome-based quality. Expand these with domain-specific harms, sampling assumptions, alert budgets, seasonality, label delay and an escalation policy. No single parity number or drift p-value establishes that a deployed system is safe, fair or useful.
