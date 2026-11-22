"""Authored deeper examples. Each lab is executed standalone by the builder."""
LABS={}
def add(chapter,title,explain,code,reading,task,solution,docs):
 LABS.setdefault(chapter,[]).append(dict(title=title,explain=explain,code=code.strip(),reading=reading,task=task,solution=solution,docs=docs))
add(13,'Multiclass metrics: whose mistakes count?',
'''A three-class problem needs one-versus-rest counts for each class. In the example, A has precision 2/3 and recall 2/3; B has precision 1/3 and recall 1/2; C has precision 1 and recall 1/3. Macro F1 gives A, B, and C equal importance. Weighted F1 weights each class’s F1 by its actual support (3, 2, and 3). Micro F1 pools all class decisions; for ordinary single-label multiclass classification using all classes, it equals accuracy. Weighted F1 is not generally the harmonic mean of weighted precision and weighted recall. Choose the averaging rule according to the question, not the highest result.''',
'''from sklearn.metrics import classification_report, f1_score, accuracy_score
actual = ["A", "A", "A", "B", "B", "C", "C", "C"] # Three labels with unequal counts.
predicted = ["A", "A", "B", "B", "A", "C", "B", "B"] # Deliberately make several error types.
print(classification_report(actual, predicted, labels=["A", "B", "C"], digits=3, zero_division=0)) # Read each class as positive in turn.
for averaging in ["macro", "micro", "weighted"]: # Different aggregation questions.
    print(averaging, round(f1_score(actual, predicted, average=averaging), 3)) # Compare the same predictions.
assert f1_score(actual, predicted, average="micro") == accuracy_score(actual, predicted) # Single-label all-class identity.
''',
'''The report’s support column counts actual examples. Precision is purity among predictions of a class, while recall is coverage of its actual examples. Four of eight labels are correct, so micro F1 and accuracy are 0.5. Macro and weighted F1 differ because the supports are unequal. The report is a description of these fixed predictions, not evidence of a fitted model’s future quality.''',
'Recalculate class B precision, recall and F1 by hand. Then change one C→B mistake to C→C and predict which class-specific metrics will change.',
'Class B has TP=1, FP=2, FN=1: precision=1/3, recall=1/2, F1=0.4. Correcting a C→B error reduces B false positives and C false negatives, affecting both classes.',
'https://scikit-learn.org/1.8/modules/model_evaluation.html#classification-metrics')
add(31,'One-versus-rest ROC and precision–recall',
'''For each class, compare that class against all remaining classes. The classifier’s probability columns must match classes_. Turn y into an indicator table with label_binarize using that same order. ROC and precision–recall curves take scores, not already thresholded labels. Macro averages give equal weight to classes; prevalence affects precision–recall behavior. The number of curve thresholds can differ because thresholds are based on distinct scores. Do not average ROC AUCs by passing an arbitrary label order or report a class with no positive validation examples.''',
'''from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
X, y = load_wine(return_X_y=True) # A three-class task.
Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, random_state=42) # Development-only evaluation split.
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xt, yt) # Fit without validation rows.
p = model.predict_proba(Xv) # Columns follow model.classes_.
Y = label_binarize(yv, classes=model.classes_) # Align one-versus-rest targets.
for j, label in enumerate(model.classes_):
    fpr, tpr, thresholds = roc_curve(Y[:, j], p[:, j]) # Class j versus all others.
    precision, recall, pr_thresholds = precision_recall_curve(Y[:, j], p[:, j]) # Use the same continuous scores.
    print("Class", label, "AUC", round(roc_auc_score(Y[:, j], p[:, j]), 3), "AP", round(average_precision_score(Y[:, j], p[:, j]), 3)) # Report per-class ranking.
print("Macro OVR AUC:", round(roc_auc_score(yv, p, multi_class="ovr", average="macro", labels=model.classes_), 3)) # Explicit strategy and class order.
assert Y.shape == p.shape # Every probability column has a matching target.
''',
'''Each class receives its own AUC and average precision. AUC summarizes ROC ranking; AP summarizes precision across recall changes. The example compares classes, not thresholds for deployment. Do not tune a threshold for each class and claim mutually exclusive labels without defining how conflicting decisions are resolved.''',
'Add a multilabel classification_report using the original Chapter 31 indicator targets. Compare micro, macro, weighted and samples F1.',
'For multilabel tasks, samples averaging computes a score per example before averaging. Micro aggregates all label decisions. Subset accuracy still requires the entire predicted set to match.',
'https://scikit-learn.org/1.8/modules/model_evaluation.html#multiclass-and-multilabel-classification')
add(14,'Compare imputers and preserve missingness',
'''A median is a simple, auditable reference. KNNImputer borrows from rows close in their observed values, so scale and meaningful distances matter. IterativeImputer models a missing column using other columns repeatedly; it is experimental and requires an explicit enable import in this frozen version. It does not recover missing ground truth or solve missing-not-at-random mechanisms. MissingIndicator can explicitly record missingness; features="all" creates an indicator for every input column so new missingness does not change width. Combine indicators with imputed values, rather than treating indicator-only output as the original features.''',
'''import numpy as np
from sklearn.experimental import enable_iterative_imputer # Explicit experimental opt-in before importing the class.
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator
train = np.array([[1., 10.], [2., np.nan], [3., 30.], [4., 40.], [5., 50.]]) # Correlated demonstration features.
new = np.array([[np.nan, 25.], [3.5, np.nan]]) # Two future rows with different missingness.
for imputer in [SimpleImputer(strategy="median"), KNNImputer(n_neighbors=2), IterativeImputer(max_iter=20, random_state=42)]:
    transformed = imputer.fit(train).transform(new) # Only training data teaches the imputation rule.
    print(type(imputer).__name__, transformed.round(2).tolist()) # Compare assumptions through their output.
    assert transformed.shape == new.shape and np.isfinite(transformed).all() # Check the representation.
flags = MissingIndicator(features="all").fit(train).transform(new) # Stable indicator width for all columns.
print("Missing flags:", flags.astype(int).tolist()) # One indicates an originally absent value.
''',
'''Different filled values reflect different assumptions. This deliberately tiny example compares behavior, not predictive accuracy. An indicator value of one records absence, while the imputed measurement is an estimate. Evaluate the complete imputer-plus-model pipeline on suitable development folds before choosing a method.''',
'Compare median and KNN imputation inside a scaled pipeline on a documented development dataset. Record runtime as well as score. Investigate whether missingness itself is predictive.',
'Fit preprocessing in each training fold. Keep a median reference and justify the distance representation before choosing KNN. Missingness can be informative without being causally meaningful.',
'https://scikit-learn.org/1.8/modules/impute.html')
add(16,'Choose a transformation for a reason',
'''RobustScaler centers by the median and scales by an interquartile range; outliers still exist afterward. PowerTransformer with Yeo–Johnson can transform positive, zero and negative values; Box–Cox requires strictly positive values. QuantileTransformer learns an empirical ranking map and can reshape marginals but distort distances and flatten unseen extremes at the learned bounds. KBinsDiscretizer replaces continuous values with bins; the strategy determines how boundaries are chosen. SplineTransformer constructs smooth piecewise-polynomial basis features. More sophisticated transformations add assumptions and parameters, so evaluate them in complete pipelines.''',
'''import numpy as np
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer, KBinsDiscretizer, SplineTransformer
X = np.array([[-3.], [-1.], [0.], [1.], [2.], [4.], [8.], [50.]]) # Include skew and a large extreme.
transformers = [RobustScaler(), PowerTransformer(method="yeo-johnson"), QuantileTransformer(n_quantiles=8, output_distribution="normal", random_state=42), KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform"), SplineTransformer(n_knots=3, degree=2, include_bias=False)] # Explicit bounded configurations.
for transformer in transformers:
    Z = transformer.fit_transform(X) # Illustrate the representation on this small training table.
    print(type(transformer).__name__, "shape", Z.shape, "first row", Z[0].round(3).tolist()) # Inspect width and scale.
    assert Z.shape[0] == len(X) and np.isfinite(Z).all() # Preserve examples and numerical validity.
''',
'''The first three mappings retain one input column; discretization assigns bin numbers and splines create multiple basis features. Ordinal bin numbers impose an order; one-hot bins are a different representation. These training transforms are not an evaluation. A transformed normal marginal does not make the joint data distribution normal.''',
'Compare a linear model with raw, power-transformed, and spline-expanded input using development cross-validation. Plot the mappings before interpreting a better score.',
'Put each transformer before the estimator in a Pipeline, use identical folds, and inspect prediction behavior beyond the observed range. Do not choose a transformation using the final test.',
'https://scikit-learn.org/1.8/modules/preprocessing.html')
add(17,'Inspect named features without changing global state',
'''Named outputs make pipeline debugging concrete: inspect which columns each branch produced and whether a target or identifier slipped in. set_config(transform_output="pandas") changes the global default. config_context applies the same option only within a block and restores the previous setting. Sparse one-hot output is incompatible with a pandas output request in this configuration; explicitly choose dense output only when its memory cost is acceptable. Keep transformations inside the evaluation boundary even while inspecting names.''',
'''import pandas as pd
from sklearn import config_context
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
X = pd.DataFrame({"hours": [1., 2., 3.], "city": ["A", "B", "A"]}) # Labeled raw inputs.
prep = ColumnTransformer([("num", StandardScaler(), ["hours"]), ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["city"])]) # Two named branches.
with config_context(transform_output="pandas"): # Scope the output setting instead of changing the whole session.
    Z = prep.fit_transform(X) # Inspect training output in a named table.
    print("Columns:", Z.columns.tolist()) # Verify column names and order.
    print("Names agree:", Z.columns.tolist() == prep.get_feature_names_out().tolist()) # Compare both inspection APIs.
assert len(prep.get_feature_names_out()) == 3 # One scaled feature and two category indicators.
''',
'''Prefixes identify the preprocessing branch. DataFrame outputs assist inspection, but input dtypes and category meanings still need checking. Returning names is not proof the model’s inputs were available at prediction time.''',
'Inspect a fitted mixed-data pipeline and list the raw input columns, transformed feature names, and final estimator width.',
'Use feature_names_in_ where available, get_feature_names_out on the preprocessing step, and n_features_in_ on the final estimator. Avoid calling fit on held-out records just to inspect them.',
'https://scikit-learn.org/1.8/auto_examples/miscellaneous/plot_set_output.html')
add(22,'Native categorical histogram boosting',
'''HistGradientBoosting can recognize pandas categorical columns with categorical_features="from_dtype". A category column describes a vocabulary, not a ranked number. Convert relevant columns to dtype category; nullable numeric Float64 remains numeric, not automatically categorical. Cardinality must respect the estimator’s bin limits. Unknown categories are treated as missing by the categorical prediction path. Native handling avoids a one-hot expansion but does not remove the need for a correct split, feature availability checks, or schema validation.''',
'''import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(42) # Reproducible example rows.
X = pd.DataFrame({"city": pd.Categorical(np.tile(["A", "B", "C"], 60)), "value": pd.array(rng.normal(size=180), dtype="Float64")}) # Explicit category and nullable numeric columns.
y = ((X["city"] == "A") | (X["value"] > 0.5)).astype(int) # Construct an educational target.
Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, random_state=42) # Development validation split.
model = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=30, min_samples_leaf=5, random_state=42).fit(Xt, yt) # Learn native category splits.
print("Categorical mask:", model.is_categorical_.tolist()) # Inspect recognized feature types.
print("Validation accuracy:", round(model.score(Xv, yv), 3)) # Evaluate unseen rows.
assert model.is_categorical_.tolist() == [True, False] # The nullable numeric column stays numeric.
''',
'''The boolean mask identifies city as categorical and value as numerical. This constructed relationship is deliberately easy. Compare native handling and a one-hot pipeline on representative development data with equivalent budgets before deciding which is better.''',
'Add a prediction record with an unseen city category and a missing numeric value. Compare its behavior with your one-hot pipeline.',
'Build the raw DataFrame with an explicit category dtype and preserve training schema. Check validity and output shape; a successful prediction does not establish reliable behavior for an unseen population.',
'https://scikit-learn.org/1.8/modules/ensemble.html#categorical-support-in-gbdt')
add(22,'Regression prediction intervals with quantile loss',
'''A point prediction describes one location of the target distribution. Quantile models estimate conditional lower and upper quantiles. An interval from the 0.1 and 0.9 quantiles has a nominal central coverage of 80%, but fitted models do not automatically achieve that coverage. Evaluate both coverage and interval width on held-out development data and inspect subgroups. Quantile crossing can occur when independently fitted lower and upper models disagree. A bootstrap of fitted mean predictions alone measures model-fit variability, not the full distribution of future outcomes: observation noise is missing.''',
'''import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
rng = np.random.default_rng(42)
X = rng.uniform(0, 5, size=(300, 1)) # A numerical predictor.
y = 2 * X[:, 0] + rng.normal(size=300) * (0.5 + X[:, 0]) # Noise grows with the predictor.
Xt, Xv, yt, yv = train_test_split(X, y, random_state=42) # Validate on unseen development examples.
lower_model = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=50, max_depth=2, random_state=42).fit(Xt, yt) # Estimate a lower conditional quantile.
upper_model = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=50, max_depth=2, random_state=42).fit(Xt, yt) # Estimate an upper conditional quantile.
lo, hi = lower_model.predict(Xv), upper_model.predict(Xv) # Bounds for each validation row.
print("Empirical coverage:", round(np.mean((yv >= lo) & (yv <= hi)), 3)) # Fraction of outcomes within their intervals.
print("Mean signed width:", round(np.mean(hi - lo), 3)) # Width in target units; inspect crossings separately.
print("Crossed bounds:", int((lo > hi).sum())) # Do not hide invalid intervals.
assert lo.shape == hi.shape == yv.shape # One bound pair per example.
''',
'''Coverage describes these held-out outcomes. It can differ from 0.8 through estimation error, limited data, and distribution mismatch. Wide intervals can cover many outcomes without being useful. A narrow confidence interval for average MAE is a different object from a prediction interval for one future target.''',
'Compare nominal 80% and 90% quantile intervals using held-out coverage, width, and crossings. Report the noise assumptions and target units.',
'Use quantile pairs (0.1,0.9) and (0.05,0.95), tune only on development data, and report empirical coverage without promising conditional coverage for every subgroup.',
'https://scikit-learn.org/1.8/auto_examples/ensemble/plot_gradient_boosting_quantile.html')
add(24,'Nested cross-validation: evaluate the selection procedure',
'''A search’s best_score_ is the result selected from several noisy candidates. Reporting it as an unbiased post-selection estimate is optimistic. Nested cross-validation puts the search inside an outer evaluation loop: the inner folds choose settings using outer-training data only, and the outer validation fold evaluates that complete choice. This is one valid design; a genuinely untouched final test set can also evaluate a frozen tuned workflow. Neither design fixes leakage or an inappropriate split. Different outer folds may select different settings, so nested CV evaluates the procedure, not one universal hyperparameter value.''',
'''from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
X, y = load_wine(return_X_y=True) # Bounded multiclass development data.
inner = StratifiedKFold(3, shuffle=True, random_state=1) # Select settings within each outer-training partition.
outer = StratifiedKFold(3, shuffle=True, random_state=2) # Evaluate the complete selection procedure.
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)) # Keep scaling inside both boundaries.
search = GridSearchCV(pipe, {"logisticregression__C": [0.1, 1.]}, cv=inner, scoring="accuracy", refit=True) # Two candidates, three inner folds.
scores = cross_val_score(search, X, y, cv=outer, scoring="accuracy", error_score="raise") # Each outer validation partition stays out of tuning.
print("Outer accuracies:", scores.round(3)) # Three evaluation results.
print("Mean / spread:", round(scores.mean(), 3), round(scores.std(), 3)) # Descriptive fold summary, not a confidence interval.
print("Fits:", 3 * (2 * 3 + 1)) # Three searches with six candidate fits and one refit each.
assert scores.shape == (3,) # One independent role per outer validation fold.
''',
'''There are 21 fits in this bounded design. Fold standard deviation is descriptive because the training sets overlap. After evaluating the process, a final search can fit a deployable model on the permitted development data; the outer scores are not scores of that final fitted instance.''',
'Explain where the scaler, search, and final estimator fit in one outer round. Compare this design with a single reserved final test after tuning.',
'All preprocessing and tuning remain within outer training. A reserved test is a valid alternative if it has not influenced any settings or threshold decisions. Do not claim nested CV is the only honest evaluation design.',
'https://scikit-learn.org/1.8/auto_examples/model_selection/plot_nested_cross_validation_iris.html')
add(27,'Learning curves and validation curves you can read',
'''A learning curve changes the training-set size while keeping the learning procedure fixed. A validation curve changes one model setting. Read training and validation together: very strong training with weaker validation suggests sensitivity or overfitting; both weak suggests mismatch or insufficiently useful inputs, but neither pattern proves a unique diagnosis. More data is promising when validation improves with size, not merely because training accuracy is high. Curves are development diagnostics and must not consume a reserved final test.''',
'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve
X, y = load_wine(return_X_y=True)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)) # Refit preprocessing for every curve point.
cv = StratifiedKFold(3, shuffle=True, random_state=42)
sizes, train, val = learning_curve(pipe, X, y, cv=cv, train_sizes=[0.4, 0.7, 1.0], scoring="accuracy", shuffle=True, random_state=42) # Vary training size.
values = [0.01, 0.1, 1., 10.]
vtrain, vval = validation_curve(pipe, X, y, param_name="logisticregression__C", param_range=values, cv=cv, scoring="accuracy") # Vary regularization setting.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(sizes, train.mean(axis=1), label="Training") # Average one result per fold at each size.
axes[0].plot(sizes, val.mean(axis=1), label="Validation")
axes[0].set(xlabel="Training examples", ylabel="Accuracy", title="Learning curve")
axes[1].semilogx(values, vtrain.mean(axis=1), label="Training")
axes[1].semilogx(values, vval.mean(axis=1), label="Validation")
axes[1].set(xlabel="C (inverse regularization strength)", ylabel="Accuracy", title="Validation curve")
for ax in axes: ax.legend()
fig.tight_layout(); fig.savefig("curves.png"); plt.close(fig) # Save both diagnostic plots.
print("Sizes:", sizes.tolist())
print("Learning validation means:", val.mean(axis=1).round(3).tolist())
print("Setting validation means:", vval.mean(axis=1).round(3).tolist())
assert train.shape == val.shape == (3, 3) # Three sizes and three folds.
''',
'''Each printed validation mean averages folds at a size or C setting. The graphs expose gaps hidden by a single average. These small Wine curves can have sampling noise and nonmonotonic points. Do not call a curve an unbiased final estimate after using it to select settings.''',
'Repeat with a shallow and a deep tree. Explain one hypothesis suggested by each curve and one alternative explanation.',
'Look at both levels and gaps. A small training–validation gap at poor performance can suggest underfitting, but poor data or label noise can produce similar patterns. Confirm with controlled comparisons.',
'https://scikit-learn.org/1.8/modules/learning_curve.html')
add(29,'Clustering quality and stability are different questions',
'''Silhouette and Calinski–Harabasz are higher-is-better geometric summaries; Davies–Bouldin is lower-is-better. Their assumptions can favor compact separated groups and may poorly reflect density or nonconvex clustering. Stability asks whether reasonable changes in initialization or samples preserve a grouping. Adjusted Rand compares two assignments without requiring identical numeric cluster IDs. For a shared-row stability check it needs labels for the same examples; comparing arbitrary row orders is meaningless.''',
'''from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
X = StandardScaler().fit_transform(load_wine().data) # Exploratory representation.
a = KMeans(3, n_init=10, random_state=1).fit_predict(X)
b = KMeans(3, n_init=10, random_state=9).fit_predict(X) # Change initialization on the same rows.
print("Silhouette:", round(silhouette_score(X, a), 3))
print("Davies-Bouldin:", round(davies_bouldin_score(X, a), 3))
print("Calinski-Harabasz:", round(calinski_harabasz_score(X, a), 3))
print("Seed agreement (ARI):", round(adjusted_rand_score(a, b), 3)) # Compare assignments invariant to cluster numbering.
assert len(a) == len(b) == len(X) # Same examples in the same order.
''',
'''A high seed agreement says the initialization change produced similar labels on this table; it does not establish stability to new samples or domain changes. A larger Calinski–Harabasz score is not automatically comparable across unrelated datasets. If a method returns only one cluster, several geometric scores are undefined.''',
'Compare several K values using all three metrics, then repeat with resampled training subsets and evaluate assignments on a common reference set.',
'Keep preprocessing consistent and match rows before computing ARI. Record cluster sizes and whether high scores correspond to useful groups rather than selecting the largest metric blindly.',
'https://scikit-learn.org/1.8/modules/clustering.html#clustering-performance-evaluation')
add(30,'Four selectors inside the evaluation boundary',
'''SelectKBest scores individual features; correlated or interacting features can challenge univariate ranking. SelectFromModel retains features using a fitted estimator’s importance or coefficients. RFE recursively removes low-ranked features and refits. SequentialFeatureSelector adds or removes features based on internal cross-validation. These wrappers can be expensive and their selections can vary with the sample. Any selection using y must remain inside the outer evaluation pipeline.''',
'''from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
X, y = make_classification(n_samples=120, n_features=6, n_informative=3, random_state=42) # Small enough for wrapper demonstrations.
base = LogisticRegression(max_iter=1000)
selectors = [SelectKBest(f_classif, k=3), SelectFromModel(base, threshold="median"), RFE(base, n_features_to_select=3), SequentialFeatureSelector(base, n_features_to_select=3, cv=2)]
cv = StratifiedKFold(3, shuffle=True, random_state=42)
for selector in selectors:
    pipe = make_pipeline(StandardScaler(), selector, LogisticRegression(max_iter=1000)) # Fit scaling and selection on training folds only.
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(type(selector).__name__, round(scores.mean(), 3))
assert len(scores) == 3 # Three held-out evaluations per complete workflow.
''',
'''The scores compare workflows, not permanently discovered true feature sets. The sequential selector has an inner comparison within each outer training fold. If you use these same scores to choose the selector family, report the selected result on an untouched test or through a further nested evaluation.''',
'Record feature-selection frequency across folds and compare with an all-feature baseline. Report runtime and instability alongside score.',
'Fit on each permitted training partition, inspect get_support(), and count selected indices. Avoid treating a frequently selected feature as a cause.',
'https://scikit-learn.org/1.8/modules/feature_selection.html')
add(32,'Permutation importance, PDP, and ICE in pictures',
'''The existing example already computes permutation importance; its label “Mean importances” means mean score changes over repeated permutations, not impurity importance. This lab visualizes that variation and adds ICE. A PDP averages model response across examples, which can hide opposing individual patterns. ICE draws a response for each selected example; kind="both" overlays those curves and their mean. Both methods can create implausible feature combinations when inputs depend on each other. Neither represents a causal intervention.''',
'''import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
X, y = make_friedman1(n_samples=180, n_features=5, noise=1, random_state=42)
Xt, Xv, yt, yv = train_test_split(X, y, random_state=42) # Development inspection partition.
model = RandomForestRegressor(n_estimators=30, min_samples_leaf=3, random_state=42).fit(Xt, yt)
importance = permutation_importance(model, Xv, yv, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error") # Score loss after shuffling.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(range(5), importance.importances_mean, yerr=importance.importances_std) # Descriptive repeat variation.
axes[0].set(xlabel="Feature index", ylabel="MAE score decrease", title="Permutation importance")
PartialDependenceDisplay.from_estimator(model, Xv, [0], kind="both", method="brute", subsample=20, random_state=42, grid_resolution=12, ax=axes[1]) # Individual curves and their average.
axes[1].set_title("PDP + ICE for feature 0")
fig.tight_layout(); fig.savefig("inspection-depth.png"); plt.close(fig)
print("Permutation means:", importance.importances_mean.round(3).tolist())
assert importance.importances.shape == (5, 5) # Five features by five permutations.
''',
'''Positive permutation importance means shuffling reduced the chosen score on average. Error bars show repeat variability, not a confidence interval. ICE curves describe model response for individual feature contexts; they need not be parallel. Plotting 20 rows is a readability choice, not a representative subgroup analysis.''',
'Compare PDP and ICE for a feature interacting with another feature. Describe what averaging hides and check whether the feature combinations are plausible.',
'Use kind="both", identify heterogeneity, and inspect observed joint distributions before interpreting. Correlated substitutes can dilute individual permutation importances.',
'https://scikit-learn.org/1.8/modules/partial_dependence.html')
add(33,'Metric uncertainty, OOF predictions, and a permutation test',
'''Different uncertainty questions need different procedures. A paired bootstrap of fixed held-out (actual, prediction) pairs estimates sampling variability of a metric conditional on a fixed fitted model; it does not include retraining or tuning uncertainty. Resample entities or time blocks instead of rows when observations are dependent. cross_val_predict produces out-of-fold predictions, but a metric on pooled predictions can differ from a mean fold score, especially for non-decomposable metrics. It also does not remove selection bias if its settings were chosen using the same data. permutation_test_score asks whether the scoring procedure outperforms a label-permutation null under exchangeability; it is not a comparison between two tuned models or a causal test.''',
'''import numpy as np
from sklearn.datasets import make_regression, load_wine
from sklearn.model_selection import train_test_split, cross_val_predict, permutation_test_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
X, y = make_regression(n_samples=180, n_features=4, noise=15, random_state=42)
Xt, Xv, yt, yv = train_test_split(X, y, random_state=42)
pred = LinearRegression().fit(Xt, yt).predict(Xv) # Fixed held-out predictions for one prechosen model.
rng = np.random.default_rng(42)
errors = []
for _ in range(300):
    index = rng.integers(0, len(yv), size=len(yv)) # Resample paired rows together.
    errors.append(mean_absolute_error(yv[index], pred[index]))
print("Held-out MAE:", round(mean_absolute_error(yv, pred), 3))
print("Approximate 95% percentile interval:", np.quantile(errors, [0.025, 0.975]).round(3).tolist()) # Conditional metric uncertainty, not prediction bounds.
W, labels = load_wine(return_X_y=True)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
cv = StratifiedKFold(3, shuffle=True, random_state=42)
oof = cross_val_predict(pipe, W, labels, cv=cv, method="predict") # Each row predicted without fitting that row.
score, null_scores, pvalue = permutation_test_score(pipe, W, labels, cv=cv, n_permutations=19, random_state=42, scoring="accuracy") # Small demonstration, coarse p-value resolution.
print("OOF prediction count:", len(oof))
print("Observed score / permutation p-value:", round(score, 3), round(pvalue, 3))
assert len(oof) == len(labels) and len(null_scores) == 19
''',
'''The percentile endpoints are an approximate uncertainty interval for mean absolute error under the stated row-bootstrap assumptions. They do not bound individual future targets. With 19 permutations, p-values have resolution 1/20=0.05 using the corrected count; this is a teaching budget, not a powerful significance study. A low p-value does not quantify practical usefulness.''',
'Write a report paragraph that clearly separates a point metric, an approximate conditional metric interval, and a prediction interval. Explain when grouped resampling is necessary.',
'Name what was held fixed, what was resampled, the assumed sampling unit, and which sources of uncertainty were omitted. Avoid declaring superiority from non-overlapping heuristic fold error bars.',
'https://scikit-learn.org/1.8/modules/cross_validation.html')
add(33,'Subgroup checks and fairness questions',
'''Aggregate quality can hide unequal error rates. Before comparing groups, define the target, the decision, group membership, the harms of false positives and false negatives, and whether the data represents each group. Report counts and support alongside metrics. Small groups produce noisy estimates; intersectional groups can be especially sparse. Equal recall, equal false-positive rates, equal calibration, and equal selection rates are distinct goals and can conflict. A numerical parity check is not a fairness certificate, and sensitive features can still be proxied by other columns after removal.''',
'''import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score
actual = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
predicted = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
groups = np.array(["A"] * 6 + ["B"] * 6) # Illustrative groups, not a claim about real populations.
for group in ["A", "B"]:
    mask = groups == group
    tn, fp, fn, tp = confusion_matrix(actual[mask], predicted[mask], labels=[0, 1]).ravel() # Explicit labels preserve matrix shape.
    print(group, "n", int(mask.sum()), "positives", int(actual[mask].sum()), "recall", round(recall_score(actual[mask], predicted[mask], zero_division=0), 3), "FPR", round(fp / (fp + tn), 3))
assert len(groups) == len(actual) == len(predicted)
''',
'''The six-row groups are too small for strong conclusions. The example demonstrates reporting counts and denominators. A zero denominator needs explicit handling and should not be hidden as a good score. Real fairness analysis also requires context, affected people’s needs, label-quality investigation, and a justified decision policy.''',
'Add subgroup counts and error types to an earlier project report. Name one unmeasured group or harm that your analysis cannot evaluate.',
'Report descriptive results with uncertainty and privacy-aware grouping. Do not claim removing a protected feature eliminates discrimination or that matching one rate proves fairness.',
'https://scikit-learn.org/1.8/modules/model_evaluation.html')
add(34,'Feature names and version compatibility are part of the artifact',
'''A fitted estimator can record feature_names_in_ when trained on string-named DataFrame columns. Predicting with an unnamed array may emit a warning; supplying named columns in the wrong order can raise an error. Do not silence those signals without checking semantics. Use public estimator behavior rather than relying on a private function named __sklearn_check_feature_names. Record versions and compare known test-record predictions after loading; pickle/joblib formats do not promise arbitrary cross-version compatibility. If source or dependencies changed, retraining in the target environment may be safer than forcing a load.''',
'''import warnings
import pandas as pd
from sklearn.linear_model import LinearRegression
X = pd.DataFrame({"hours": [1., 2., 3., 4.], "distance": [3., 5., 4., 8.]})
y = [4., 7., 8., 12.]
model = LinearRegression().fit(X, y) # Fit with explicit feature names.
print("Expected names:", model.feature_names_in_.tolist())
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.predict(X.to_numpy()) # Unnamed input loses an important contract check.
    print("Unnamed-input warnings:", len(caught))
try:
    model.predict(X[["distance", "hours"]]) # Named columns appear in a different order.
except ValueError:
    print("Reordered named columns: rejected")
assert model.feature_names_in_.tolist() == ["hours", "distance"]
''',
'''The warning is useful evidence of a missing input contract. An array with reversed columns could still produce a number without telling you it is semantically wrong. Preserve names and validate units, not just the number of columns. Never load an untrusted pickle-based artifact to inspect what is inside it.''',
'Add a model manifest containing feature names, input units, package versions, training-data reference, and fixed record/prediction checks.',
'Validate before prediction and refuse unsupported schemas. Store the manifest beside the artifact. A recorded version is documentation, not a guarantee of safe loading in another version.',
'https://scikit-learn.org/1.8/model_persistence.html')
add(35,'Track an experiment and identify its data',
'''An experiment record connects settings, split definitions, code version, input data identity, environment, metrics and artifacts. A hash identifies exact bytes; it does not preserve those bytes or authorize sharing them. Git is useful for code, while data stores or a tool such as DVC can preserve permitted dataset versions. A container captures part of the environment but does not freeze the dataset, host hardware, remote services or all randomness. Begin with a transparent local record, then move to MLflow when searching and comparing many runs becomes cumbersome.''',
'''import hashlib
import json
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_wine(return_X_y=True)
Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000)).fit(Xt, yt)
record = {"dataset": "sklearn.load_wine", "data_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(), "split_seed": 42, "parameters": {"C": 1}, "validation_accuracy": model.score(Xv, yv), "code_revision": "record-your-actual-git-commit"} # Replace the explicit code marker in a real project.
Path("experiment.json").write_text(json.dumps(record, indent=2)) # Write a portable local record.
loaded = json.loads(Path("experiment.json").read_text())
print("Recorded dataset:", loaded["dataset"])
print("Hash length:", len(loaded["data_sha256"]))
print("Validation accuracy:", round(loaded["validation_accuracy"], 3))
assert loaded["parameters"]["C"] == 1
''',
'''The 64-character SHA-256 string fingerprints these array bytes. Record shape, dtype and source too when moving beyond this demonstration. The local JSON is an experiment record, not a fully managed tracker. A real record needs the actual code revision and package manifest instead of the illustrative marker.''',
'Add a run manifest to an earlier project. Then follow the optional tooling guide to log the same parameters and metric in MLflow and version a permitted data file with DVC.',
'Keep raw sensitive or restricted data out of public repositories. Record a durable access reference. Use the official MLflow, DVC and Docker guides linked in Resources; those integrations have separate installation and operational needs.',
'https://mlflow.org/docs/latest/tracking/')
add(39,'A small neural-network model-family tour',
'''MLPClassifier and MLPRegressor are feed-forward neural networks trained through backpropagation. hidden_layer_sizes defines layer widths; activation defines nonlinear response; alpha penalizes weights. Numerical scaling matters for optimization. Regression targets with a very large scale may also need a target transformation. Scikit-learn MLPs are useful for learning and bounded tabular experiments; they do not replace a full deep-learning framework or imply GPU training. Avoid assuming that adding neurons improves an already appropriate tree or linear baseline.''',
'''from sklearn.datasets import make_moons, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
X, y = make_moons(n_samples=180, noise=0.2, random_state=42)
Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, random_state=42)
classifier = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(12,), solver="lbfgs", max_iter=1000, alpha=0.1, random_state=42)).fit(Xt, yt) # Scale before fitting a bounded network.
print("MLP validation accuracy:", round(classifier.score(Xv, yv), 3))
R, t = make_regression(n_samples=120, n_features=3, noise=5, random_state=42)
Rt, Rv, tt, tv = train_test_split(R, t, random_state=42)
regressor = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=1000, alpha=0.1, random_state=42)), transformer=StandardScaler()).fit(Rt, tt) # Also scale the regression target, reversing it for prediction.
print("MLP original-unit MAE:", round(mean_absolute_error(tv, regressor.predict(Rv)), 3))
assert len(regressor.predict(Rv)) == len(tv)
''',
'''The classifier and regressor solve different tasks on different datasets, so their two scores cannot be compared. lbfgs is a bounded small-data solver choice here. Convergence depends on scaling, iteration budget and settings; a warning should be investigated rather than suppressed or confused with a performance guarantee.''',
'Compare a scaled MLP with a logistic baseline on moons, then with a tree model. Report validation behavior and fit time.',
'Use the same partitions and tune only within development data. Record hidden widths, alpha, solver, seed and any convergence warnings.',
'https://scikit-learn.org/1.8/modules/neural_networks_supervised.html')
add(43,'Hard voting, soft voting, AdaBoost, and stacking',
'''Hard voting combines class decisions; soft voting combines probabilities and is sensitive to their calibration. AdaBoost sequentially reweights training emphasis, unlike independently bagged trees. Stacking trains a final estimator on out-of-fold predictions from base learners. passthrough=True additionally supplies original features to the final estimator; scale them there when needed. If preprocessing sits outside the stacking cross-fit, the meta-features may see preprocessing learned using its validation rows. Keep learned preprocessing inside each base pipeline, and keep final-estimator preprocessing within its own training workflow.''',
'''from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, AdaBoostClassifier
X, y = load_wine(return_X_y=True)
Xdev, Xtest, ydev, ytest = train_test_split(X, y, stratify=y, random_state=42) # Keep final-test examples reserved.
base = [("linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))), ("tree", DecisionTreeClassifier(max_depth=3, random_state=42))]
inner = StratifiedKFold(3, shuffle=True, random_state=1)
outer = StratifiedKFold(3, shuffle=True, random_state=2)
models = {"hard vote": VotingClassifier(base, voting="hard"), "soft vote": VotingClassifier(base, voting="soft"), "AdaBoost": AdaBoostClassifier(n_estimators=25, random_state=42), "stacking": StackingClassifier(base, final_estimator=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), cv=inner, passthrough=True)} # Separate base and meta-fitting boundaries.
for name, model in models.items():
    score = cross_val_score(model, Xdev, ydev, cv=outer, scoring="accuracy") # Compare full unfitted workflows.
    print(name, round(score.mean(), 3))
assert len(score) == 3
''',
'''The outer scores compare complete workflows; stacking’s internal folds create meta-training features without fitting a row’s base predictor on that row. Raw features also reach the scaled final pipeline because passthrough is true. Soft voting does not automatically improve calibration, and the strongest individual baseline still needs comparison.''',
'Compare all four ensembles with the individual base estimators. Repeat stacking with passthrough=False and report cost alongside score.',
'Preserve outer splits. If the comparison chooses an ensemble, use separate final evaluation before making a final performance claim. Ensembles can be less useful when their errors are highly similar.',
'https://scikit-learn.org/1.8/modules/ensemble.html')
add(47,'Monitor inputs without pretending you know accuracy',
'''Monitoring has several layers: schema and range violations, missingness, latency, failure rates, input-distribution changes, and outcome-based quality. A two-sample Kolmogorov–Smirnov statistic measures the largest gap between empirical cumulative distributions of one numerical feature. Its p-value depends on sample size and assumptions, so a tiny p-value alone is not an operational alert policy. Repeated testing across many features increases false alarms. Define windows, sample counts, practical effect thresholds, seasonal references and investigation steps before observing an incident. Changes in y given X can occur with little visible input drift, and input drift can occur without worse predictions.''',
'''import numpy as np
from scipy.stats import ks_2samp
rng = np.random.default_rng(42)
reference = rng.normal(0, 1, 200) # A retained training-era reference window.
current = rng.normal(0.6, 1, 200) # An illustrative shifted input window.
result = ks_2samp(reference, current) # Compare one numerical marginal.
print("Window sizes:", len(reference), len(current))
print("KS effect statistic:", round(result.statistic, 3))
print("KS p-value:", round(result.pvalue, 4))
print("Missing rate:", round(np.isnan(current).mean(), 3))
assert 0 <= result.statistic <= 1
''',
'''The statistic describes an input change, not predictive degradation. No target outcomes were supplied, so no accuracy claim is possible. A practical monitor logs version, timestamps, counts, schema-failure rate, latency and the outcome-delay policy. Compare labeled predictions later using representative labels and subgroup checks.''',
'Write a monitor specification for an earlier project: reference window, current window, minimum sample size, practical alert rule, responsible action and outcome-based follow-up.',
'Include a false-alarm and seasonality policy. Start with descriptive checks, review alerts before automatic retraining, and preserve rollback and version records.',
'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html')
CHECKS={
13:[('Rare classes should count as much as common classes in a multiclass F1 summary. Which averaging rule matches that goal?','Macro F1 gives each class equal weight.','Weighted F1 always gives rare classes equal weight.')],
14:[('A new input is missing in a column that was never missing during fit. You need one indicator per column. Which configuration matches that contract?','MissingIndicator(features="all") creates a stable indicator for every input column.','SimpleImputer(add_indicator=True) always creates an indicator for every original column.')],
16:[('A feature has meaningful zero and negative values. Which power transformation can handle that domain?','Yeo–Johnson can handle zero and negative values.','Box–Cox accepts arbitrary negative values without changes.')],
17:[('You need named transformer output for one inspection without changing the notebook’s global default. What should you use?','A config_context(transform_output="pandas") block scopes the setting.','Change the final-test data into an array and refit all preprocessing.')],
22:[('A quantile model estimates the 0.1 and 0.9 conditional quantiles. What can you claim before evaluating coverage?','It targets an 80% central interval, but actual held-out coverage and width still need checking.','It guarantees 80% coverage for every subgroup automatically.'),('A pandas column has nullable Float64 dtype. Does native categorical boosting automatically treat it as categorical?','No. Nullable numeric remains numeric; use an explicit categorical dtype for a category.','Yes. Every nullable dtype is a category.')],
24:[('A report presents GridSearchCV.best_score_ as the final unbiased performance after tuning. What should replace that claim?','Use outer evaluation of the search procedure or a genuinely untouched final test for the frozen tuned workflow.','Increase the grid size until best_score_ stops changing.')],
27:[('Training accuracy is high while validation accuracy is weaker across learning-curve sizes. What is justified?','This suggests sensitivity or overfitting worth investigating, not proof of a unique cause.','This proves that no amount of new data can ever help.')],
29:[('Two K-means runs assign different numeric IDs to otherwise identical groups. How should stability be compared?','Use a label-permutation-invariant comparison such as adjusted Rand on matched rows.','Compare numeric label equality directly because cluster 0 always has a fixed meaning.')],
30:[('SequentialFeatureSelector performs its own validation. Can it be fitted before the outer evaluation split?','No. Its full fitting process belongs within the outer training partition.','Yes. Internal validation makes all outer leakage impossible.')],
31:[('A one-versus-rest ROC curve takes predict_proba output. How are columns matched to classes?','Use the fitted classes_ ordering consistently when binarizing the targets.','Sort class names independently and assume the probability order matches.')],
32:[('ICE curves vary substantially while their PDP is almost flat. What can that indicate?','Individual response patterns can average out; inspect heterogeneity and feature dependence.','The model must ignore that feature for every example.')],
33:[('You bootstrap pairs of fixed held-out targets and predictions. What uncertainty is being summarized?','Conditional metric sampling variability under the resampling assumptions, not all retraining uncertainty or individual prediction bounds.','A guaranteed interval for every future prediction.'),('Two groups have equal recall. Does that establish fairness?','No. Recall is one criterion; other harms, rates, sample sizes and context still matter.','Yes. Equal recall guarantees all forms of fairness.')],
34:[('A model fitted on named columns warns when predicting an unnamed array. What should happen?','Validate the expected names, order and units before accepting the prediction input.','Suppress the warning and assume column semantics match.')],
35:[('A dataset hash is recorded but the actual data is lost. Can the hash restore it?','No. A hash identifies bytes but does not preserve them; a permitted versioned data store is also needed.','Yes. SHA-256 encodes the entire dataset reversibly.')],
39:[('MLPRegressor fits targets with a very large numerical scale. Which workflow is a reasonable candidate?','Evaluate input scaling and a target-scaling wrapper that reverses predictions into original units.','Increase hidden units until the training loss is zero and stop evaluating.')],
43:[('Stacking uses passthrough=True and a linear final model. What additional data reaches the final stage?','Original features are combined with base prediction features, so final-stage scaling may matter.','The final-test labels are automatically supplied as passthrough features.')],
47:[('A drift test has a tiny p-value after collecting millions of rows. What else is needed before acting?','Assess practical effect size, multiple testing, seasonality, sample windows and outcome-based follow-up.','Declare accuracy lost by the same amount as the p-value.')]
}
