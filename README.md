# Scikit-learn: From Foundations to Independent Projects

A learning website for building practical machine-learning skills with scikit-learn through clear lessons, worked examples, tests, and projects.

## Project status

The curriculum and teaching brief have been supplied. Website implementation and lesson development are pending. This repository currently establishes the project documentation; no live course deployment is available yet.

## Learning outcomes

Learn to inspect data, prepare it without leakage, establish baselines, build and compare models, evaluate results, improve a workflow, explain its limitations, and reproduce predictions from a saved pipeline.

## Course structure

The supplied curriculum defines 10 core modules containing 38 chapters, one optional module containing 9 extensions, and a separate 14-day project challenge.

| Module | Chapters | Coverage |
| --- | --- | --- |
| 1. First success and code vocabulary | 1–3 | First classification model, reading scikit-learn syntax, working environment |
| 2. Data handling and visualization | 4–6 | NumPy, pandas, plotting, data inspection |
| 3. Safe evaluation from the beginning | 7–9 | Splitting, leakage, pipelines, cross-validation |
| 4. Regression and classification | 10–13 | Linear and logistic regression, metrics, residuals, confusion matrices |
| 5. Preparing realistic data | 14–18 | Missing values, encoding, scaling, mixed-data pipelines, target transformations |
| 6. Choosing model families | 19–22 | Regularization, neighbors, trees, forests, gradient boosting |
| 7. Reliable improvement | 23–27 | Group/time splits, tuning, scorers, weights, imbalance, calibration, diagnostics |
| 8. Text and additional learning tasks | 28–31 | Text classification, clustering, dimensionality reduction, multilabel/multioutput |
| 9. Explaining and reproducing results | 32–35 | Inspection, reporting, persistence, validated prediction workflows |
| 10. Capstone and assessment | 36–38 | Guided capstone, flawed-project diagnosis, independent assessment |
| 11. Optional extensions | 39–47 | SVMs, anomaly detection, custom transformers, incremental learning, ensembles, time-series features, external boosting, resampling, API serving and monitoring |

Modules 1–3 use the first classification workflow as their running example. Detailed regression and classification explanations follow in Module 4.

## Lesson experience

Each lesson will provide:

1. Goals, prerequisites, a plain-language explanation, an analogy, and a step plan.
2. Step-by-step code with a **Copy code** control that copies code only.
3. Separate output boxes and explanations of what the results mean.
4. Line-by-line code notes and common mistakes.
5. A recap and vocabulary table linking to fuller explanations.
6. Error messages, their meaning, and fixes.
7. Exercises covering prediction, modification, explanation, and investigation.
8. Hidden answers that explain the reasoning.
9. A completion checklist.

Examples must run from top to bottom in the frozen course environment. Published outputs must come from executed examples, and every displayed number must be explained at least once. Mathematics is introduced where it is needed.

Knowledge checks accompany every chapter. Chapters containing code also include meaningful automated checks; conceptual work uses rubrics and knowledge checks.

## Module tests and project gates

For Modules 1–9, the brief specifies:

- A 20–30-question test, planned for 45–60 minutes.
- An 80% test pass mark, with different questions on retakes.
- A module project with a written brief, dataset, deliverables, and rubric.
- Progression to the next module after passing both the test and project.
- Review guidance pointing failed topics back to their lessons.

Projects use four rubric dimensions, each scored from 0 to 3:

- Code correctness.
- Appropriate evaluation and leakage prevention.
- Clarity of explanation.
- Reproducibility.

Passing requires at least 2 on every dimension. Evaluation requirements are applied according to the task; unsupervised investigations need appropriate evaluation rather than a classification-style target.

Module 10 contains the guided capstone, diagnosis exercise, and independent final project, plus a 40-question final test with an 80% pass mark. Each optional extension has a five-question quiz and a lab and does not gate core progress.

## Planned projects and datasets

Dataset access, exact dataset identifiers/versions, and licensing must be verified before publishing the corresponding lesson.

| Module | Project |
| --- | --- |
| 1 | Repeat the classification workflow on Wine data |
| 2 | Audit Titanic data from OpenML and identify potential leakage |
| 3 | Repair a leaking breast-cancer classification notebook |
| 4 | California housing regression and handwritten-digit classification reports |
| 5 | Ames housing pipeline accepting raw records, missing values, and unseen categories |
| 6 | Compare model families on Adult census income data |
| 7 | Rare-event classification using a bounded credit-card fraud dataset |
| 8 | 20 Newsgroups text classification, Wine clustering, and generated multilabel data |
| 9 | Package a previous pipeline with validated input, prediction script, environment specification, and report |
| 10 | Guided capstone, planted-error diagnosis, and a new independent problem |

Early examples are for learning. Their scores must not be presented as evidence of suitability for real-world deployment.

## Fourteen-day project challenge

Begins after Chapter 38. Each day is planned for 2–3 hours, with dataset instructions, deliverables, a rubric, and optional stretch work.

| Day | Focus | Chapters reused |
| --- | --- | --- |
| 1 | Regression baseline and evaluation | 7–11 |
| 2 | Multiple-feature regression | 10–11, 19 |
| 3 | Mixed-data pipeline | 14–18 |
| 4 | Classification and error analysis | 12–13 |
| 5 | Imbalanced classification | 25–27 |
| 6 | Text classification | 28 |
| 7 | Cross-validated model comparison | 9, 20–22 |
| 8 | Hyperparameter tuning | 24–25 |
| 9 | Clustering | 29 |
| 10 | Dimensionality reduction | 30 |
| 11 | Multilabel or multioutput prediction | 31 |
| 12 | Group-aware evaluation | 23 |
| 13 | Saved prediction workflow | 34–35 |
| 14 | Independent bounded problem and findings report | 32–35 |

## Workload

These are the supplied curriculum's provisional estimates, including its added module-test allowance. They are not measured learner completion times.

| Component | Estimated hours |
| --- | --- |
| Core Modules 1–9, including module tests | 98–141 |
| Module 10: capstone, diagnosis, assessment | 14–24 |
| Complete core course | 112–165 |
| Optional extensions | 44–71 |
| Core plus all extensions | 156–236 |
| Separate 14-day challenge | 28–42 |

Estimates require a beginner pilot, including one complete chapter and one test, followed by a math-heavy sample. Record active time, assistance, misunderstandings, and independent transfer.

## Version and quality policy

Choose and verify a compatible stable environment at drafting, then freeze it for the edition. Record Python, NumPy, SciPy, pandas, scikit-learn, matplotlib, seaborn, joblib, and notebook-tooling versions. Optional dependencies receive separate records.

Before publication:

- Execute examples and check their displayed outputs.
- Verify dataset access and usage conditions.
- Check exercise solutions, scoring, and leakage prevention.
- Distinguish planned features from implemented features.
- Record compatibility changes and rerun affected examples when updating dependencies.

Supporting references will include a linked glossary, syntax and shape reference, model and metric guides, common-error explanations, audit and leakage checklists, solutions, project rubrics, and a report template.

## Website direction

Planned capabilities include organized lesson navigation, mobile and desktop reading, quizzes, project pages, progress tracking, search, a glossary, and downloadable practice material.

The framework, local setup commands, deployment configuration, and hosting URL will be documented when implementation is added. In-browser Python execution is not assumed by this initial brief.

## Source brief

This README summarizes the supplied PDF, **Scikit-learn: From Foundations to Independent Projects (with lessons, tests, and projects)**. The PDF is the detailed curriculum reference; this README is the repository overview.
