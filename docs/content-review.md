# Teaching-depth review, revision 2

The review was checked against the complete authored source and generated lessons, not a truncated fetch. Existing chapter numbers and browser progress keys are retained.

## Corrections to the incoming review

- Chapter 13 already explains macro/micro/weighted averages briefly; the missing piece was a worked multiclass report and ranking-curve example.
- Chapter 32 already calls `permutation_importance`. Its mean values are repeat-averaged permutation score changes, not impurity importances. ICE plotting was absent.
- Chapter 27 already fits `CalibratedClassifierCV`; it does not merely measure uncalibrated probabilities. Actual learning/validation curve examples were missing.
- Chapters 23 and 44 already use `TimeSeriesSplit`.
- Chapter 17 already calls `get_feature_names_out` and `set_output(transform="pandas")`.
- Chapter 34 already explains untrusted pickle/joblib loading and version compatibility. A concrete feature-name-contract example was missing.
- Nested CV, stacking and several preprocessing tools were described but not demonstrated. Soft voting was implemented; hard voting, stacking and AdaBoost needed worked examples.

## Added standalone applied labs

| Chapter | Addition |
| --- | --- |
| 13 | Multiclass classification report, per-class arithmetic and averaging |
| 14 | MissingIndicator, KNNImputer and experimental IterativeImputer |
| 16 | RobustScaler, PowerTransformer, QuantileTransformer, KBinsDiscretizer, SplineTransformer |
| 17 | Scoped pandas output configuration and feature-name inspection |
| 22 | Native categorical histogram boosting; quantile prediction intervals |
| 24 | Executed nested cross-validation and fit accounting |
| 27 | Learning and validation curves with plots |
| 29 | Davies–Bouldin, Calinski–Harabasz and initialization stability |
| 30 | SelectKBest, SelectFromModel, RFE and SequentialFeatureSelector within evaluation |
| 31 | One-versus-rest ROC/AP and multilabel evaluation exercise |
| 32 | Permutation importance visualization plus PDP and ICE |
| 33 | Conditional metric bootstrap, OOF predictions, permutation testing, subgroup checks |
| 34 | Named-feature warnings and input-order rejection |
| 35 | An experiment manifest with data identity and run settings |
| 39 | MLPClassifier and MLPRegressor, including target scaling |
| 43 | Hard/soft voting, AdaBoost and stacking with passthrough |
| 47 | Numerical input-drift monitoring and outcome-based follow-up policy |

Each lab has an explanation, standalone code, an executed output, line notes, interpretation, a transfer exercise and a solution approach. The chapter notebooks and notebook bundle include the labs.

## Assessment changes

- Removed both generators that asked for complete frozen lesson outputs or their first printed line.
- Added 94 executed small transfer snippets with new inputs, including arithmetic, shapes, schema checks, scoring and information-flow reasoning.
- Removed identical generic padding. Added module-specific debugging scenarios.
- Kept the published core-test mix: 8 knowledge, 6 output, 4 debugging and 2 written questions. Written answers remain explicitly self-reviewed; no independent grader is implied.
- Existing term questions remain useful vocabulary checks. These tests are not a validated mastery instrument; projects still supply the independent-application component.

## Boundaries

- Nested CV is not the only honest post-tuning evaluation. A genuinely untouched test set also evaluates a frozen tuned workflow.
- A paired bootstrap of fixed held-out errors is a conditional metric interval, not an individual prediction interval or a full retraining-uncertainty estimate.
- Quantile intervals require empirical coverage/width checks; nominal coverage is not automatically achieved.
- Input drift is not direct evidence of predictive degradation.
- The optional MLflow/DVC/Docker/SHAP/LIME guide contains separately installed follow-on exercises. Those integrations are not represented as tested dependencies of this edition.
- Added labs extend study time. Earlier workload totals are historical provisional estimates; no revised totals are presented as measured without learner pilots.
