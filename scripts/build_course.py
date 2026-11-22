from pathlib import Path
import json, subprocess, sys, os, contextlib, io, tempfile, re
ROOT=Path(__file__).resolve().parents[1]
LESSONS=[]
MODULES=[
('First success','Your first model, one understandable step at a time.',1,3,'4–6'),
('Meet your data','Arrays, tables, and the stories in your plots.',4,6,'7–10'),
('Build good habits','Safe splits, pipelines, and fair comparisons.',7,9,'8–12'),
('Predict & understand','Regression, classification, and meaningful evaluation.',10,13,'14–20'),
('Make data model-ready','Realistic preprocessing, inside one reliable workflow.',14,18,'12–18'),
('Find your model','Explore model families and their tradeoffs.',19,22,'13–19'),
('Improve with evidence','Tune, diagnose, and make better decisions.',23,27,'12–18'),
('Explore new possibilities','Words, clusters, dimensions, and multiple targets.',28,31,'12–18'),
('Share work you trust','Explain, save, validate, and reproduce.',32,35,'7–11'),
('Put it all together','A capstone, a repair mission, and your own solution.',36,38,'14–24'),
('Keep exploring','Nine optional labs for your next adventure.',39,47,'44–71')]
def L(title,idea,analogy,walk,terms,code,reading,task,solution,q,wrong,reason,error='ValueError: Input contains NaN',fix='Inspect missing values and add training-only imputation where the estimator needs it.'):
    n=len(LESSONS)+1
    module=next(i+1 for i,m in enumerate(MODULES) if m[2]<=n<=m[3])
    vocabulary=[{'term':a,'meaning':b} for a,b in [x.split('=',1) for x in terms.split('|')]]
    LESSONS.append(dict(id=n,module=module,title=title,idea=idea,analogy=analogy,walk=walk,terms=vocabulary,code=code.strip(),reading=reading,task=task,solution=solution,question=q,answer=reason,wrong=wrong,error=error,fix=fix,goal=f'Explain {title.lower()} and apply it to a small, inspectable example.',docs='https://scikit-learn.org/1.8/user_guide.html'))
CLASS='''from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_breast_cancer(return_X_y=True)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
'''
REG='''from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X, y = make_regression(n_samples=160, n_features=4, noise=12, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
L('Your first classification model',
'You will teach a model to choose between two labels using measurements. A row is one example; a column is one measurement. X contains the measurements and y contains the known labels. This first lesson is a follow-along preview: run the whole workflow before studying each tool in depth. The breast-cancer dataset is an educational dataset, and these results do not validate a medical product.',
'A practice exam helps you learn; a fresh exam checks whether you learned something reusable.',
'First inspect the shapes. Reserve some examples with a stratified split so both labels are represented. Fit a majority-class baseline using development labels. Then fit a scaling-and-logistic-regression pipeline. Accuracy is the fraction of correct predictions. This one-time preview uses its held-out set for demonstration; later work reserves a final test set until decisions are complete. Chapters 7–9 explain splitting and pipelines; Chapters 12–13 explain classification.',
'Feature=An input measurement used to make a prediction.|Target=The answer a supervised model learns to predict.|Sample=One example represented by a row of X.|Baseline=A simple reference prediction rule.',
CLASS+'''from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent") # Learn the most common development label.
baseline.fit(X_dev, y_dev) # Teach the reference rule using development data.
model.fit(X_dev, y_dev) # Fit the scaler and classifier in order.
print("Shapes:", X.shape, y.shape) # Inspect samples, features, and labels.
print("Baseline accuracy:", round(baseline.score(X_test, y_test), 3)) # Fraction correct.
print("Model accuracy:", round(model.score(X_test, y_test), 3)) # Use identical held-out rows.
print("Classes:", model.classes_) # Probability columns follow this order.
print("First probabilities:", model.predict_proba(X_test[:1]).round(3)) # One row, two classes.
assert X.shape[0] == y.shape[0] # Each input needs one matching label.
''',
'The first shape has two dimensions: number of examples and number of features. The target has one entry per example. Each accuracy lies between zero and one; multiply by 100 for a percentage. Probability columns follow classes_, not an assumed positive-class order. A larger score than the baseline is useful evidence for this split, not proof of general usefulness.',
'Repeat this workflow with load_wine. Use stratification and report both baseline and model accuracy. Explain why the probability output now has three columns.',
'Replace load_breast_cancer with load_wine in both the import and loading call. Wine has three class labels, so predict_proba returns one probability for each of three classes. Keep the scaler in the pipeline.',
'Which data should teach the majority-class baseline?', 'All labels, including the reserved test labels.', 'Only the development labels. The held-out answers must not influence the reference rule.', 'ValueError: Found input variables with inconsistent numbers of samples','Check X.shape[0] and len(y). Every input row needs a corresponding target in the same order.')
L('Reading scikit-learn code',
'Python code is easier to read when you distinguish a tool, an object made with that tool, and an operation on that object. LinearRegression is a class. Calling LinearRegression() creates an estimator object. fit is a method that learns from examples. A learned attribute such as coef_ is available after fitting; the trailing underscore is a scikit-learn convention.',
'A class is a recipe, an object is a dish made from it, and a method is an action you can perform with that dish.',
'Read from left to right: an import names where a tool comes from; parentheses call something; a dot accesses an object member. In fit(X, y), arguments are positional. In LinearRegression(fit_intercept=True), the argument is named. Assignment stores a returned result; it does not itself train a model. Inspect each object before using it.',
'Class=A definition used to construct objects.|Method=A function accessed through an object.|Keyword argument=An argument supplied using its parameter name.|Learned attribute=A fitted value stored on an estimator, usually ending in an underscore.',
'''from sklearn.linear_model import LinearRegression # Import an estimator class.
X = [[1], [2], [3]] # Three samples, one feature each.
y = [3, 5, 7] # Three corresponding target values.
model = LinearRegression(fit_intercept=True) # Construct an unfitted object.
model.fit(X, y) # Learn coefficients from the examples.
print("Weight:", model.coef_.round(2)) # Read a learned attribute.
print("Bias:", round(model.intercept_, 2)) # Read the fitted intercept.
print("Prediction:", model.predict([[4]]).round(2)) # Pass a list of new input rows.
assert len(model.predict([[4]])) == 1 # One input row gives one prediction.
''',
'The weight multiplies the input; the intercept is added afterwards. The prediction is an array even for one sample. The nested brackets preserve the required two-dimensional input shape. A printed coefficient array can contain several values when there are several features.',
'Predict for both 4 and 5 in a single call. Explain the difference between model.fit and model.fit(X, y).',
'Use model.predict([[4], [5]]). model.fit refers to the method object; model.fit(X, y) actually calls it.',
'What does creating an estimator with parentheses do?', 'It automatically trains on all available data.', 'It creates an unfitted object. Training begins when fit is called with data.', 'NotFittedError: This LinearRegression instance is not fitted yet','Call fit with training examples before reading learned attributes or making predictions.')
L('Working environment and course files',
'A reproducible environment records the tools used by the examples. A notebook cell shares state with previously executed cells, so an out-of-order run can appear to work while hiding missing steps. A script starts at the top in a fresh process. Restarting a notebook kernel and running all cells is a practical check that the lesson is complete.',
'A recipe works more reliably when you record the ingredients and start with a clean kitchen.',
'This edition runs on Python 3.12 with the package versions shown below. Install the downloaded requirements.txt in a virtual environment, or use a compatible hosted notebook. Create an environment with python -m venv .venv; activate it using your operating system’s command; then run python -m pip install -r requirements.txt. In Colab, a changed environment can require a runtime restart.',
'Environment=A particular Python interpreter and its installed packages.|Kernel=The process executing a notebook’s Python cells.|Dependency=A package needed by your code.|Pinned version=An explicitly recorded package version.',
'''import sys # Access the running Python interpreter information.
import sklearn # Import the machine-learning library.
import numpy as np # Import array tools with a short alias.
import pandas as pd # Import table tools with a short alias.
print("Python:", sys.version.split()[0]) # Show just the version number.
print("scikit-learn:", sklearn.__version__) # Print the installed library version.
print("NumPy:", np.__version__) # Print the array-library version.
print("pandas:", pd.__version__) # Print the table-library version.
assert hasattr(sklearn, "__version__") # Check that the package imported successfully.
''',
'The displayed versions describe the environment that produced this example. A different local version does not automatically make your results wrong, but investigate API differences before comparing outputs. Floating-point formatting can vary across platforms.',
'Download this chapter’s notebook. Restart your kernel, run all cells, and record your Python and scikit-learn versions.',
'Use Restart and Run All in Jupyter, or Restart session followed by Run all in Colab. Keep the version output in the submitted notebook.',
'Why restart and run all cells?', 'To keep hidden values from earlier experiments.', 'To verify that the notebook works from a fresh process with no hidden execution history.', 'ModuleNotFoundError: No module named sklearn','Install scikit-learn into the interpreter used by your notebook. The installation package name is scikit-learn; the import name is sklearn.')
L('NumPy for modeling',
'NumPy stores numerical values in arrays with a shape. In modeling, X usually has shape (n_samples, n_features). Axis 0 moves through samples; axis 1 moves through features. A one-dimensional target array has shape (n_samples,). Shape errors are often about how values are organized rather than what the values are.',
'A classroom seating chart has rows and columns. Counting students per row is different from counting students per column.',
'Create a small two-feature dataset. Select a row with X[0], a column with X[:, 0], and a two-dimensional one-column table with X[:, [0]]. The colon means all entries along that axis. reshape changes organization without adding examples. A seed makes a random generator repeatable for the same calls.',
'Array=A numerical container with a fixed shape and data type.|Axis=A direction along an array’s dimensions.|Slice=A selection using start, stop, or step positions.|Broadcasting=Rules that let compatible array shapes participate in arithmetic.',
'''import numpy as np # Load numerical array operations.
X = np.array([[1, 10], [2, 20], [3, 30]]) # Three samples with two features.
print("Shape:", X.shape) # Inspect row and column counts.
print("First row:", X[0]) # Select sample zero.
print("First feature:", X[:, 0]) # Select feature zero as a vector.
print("Column means:", X.mean(axis=0)) # Average down the sample axis.
print("One-column shape:", X[:, [0]].shape) # Preserve the feature dimension.
print("Shifted first row:", (X + [1, 100])[0]) # Broadcast one adjustment per feature.
assert X[:, [0]].shape == (3, 1) # Check the intended model-input shape.
''',
'Indexing starts at zero. Selecting X[:, 0] removes the feature dimension; X[:, [0]] keeps it. Column means summarize each feature separately. Broadcasting adds 1 to the first feature and 100 to the second for every sample.',
'Create four examples with three features. Print the shape, the second sample, the last feature, and the mean of each feature.',
'Use np.array with four inner lists of length three. Select X[1], X[:, -1], and X.mean(axis=0).',
'Which selection keeps one feature as a two-dimensional table?', 'X[:, 0]', 'X[:, [0]] preserves a column dimension; X[:, 0] produces a one-dimensional vector.', 'ValueError: Expected 2D array, got 1D array instead','For a single feature, use reshape(-1, 1) or select with a list of column indices.')
L('pandas for modeling',
'A DataFrame attaches names and types to a table. This makes it easier to separate useful features from labels and identifiers. Missing values and duplicates need investigation: an empty cell may be meaningful, and identical measurements do not always mean the same entity was recorded twice.',
'A spreadsheet becomes more useful when every column has a meaningful heading and an understood unit.',
'Inspect a small table before modeling. Select feature columns with a list of names, not by accidentally including the target. isna creates a table of missing-value flags; sum counts them per column. Keep row alignment between X and y. Use pd.read_csv(path) for a file you downloaded and inspect dtypes before assuming a column is numerical.',
'DataFrame=A labeled table with rows and columns.|Series=A labeled one-dimensional column.|Data type=The representation of values in a column.|Missing value=An unavailable value represented by a marker such as NaN.',
'''import pandas as pd # Load labeled table operations.
df = pd.DataFrame({"hours": [1, 2, None, 4], "score": [20, 35, 40, 65]}) # Build a small table.
print(df.to_string(index=False)) # Inspect the values without row labels.
print("Missing per column:") # Label the next output.
print(df.isna().sum().to_string()) # Count missing values in each column.
X = df[["hours"]] # Select inputs while preserving a table.
y = df["score"] # Select the target as a Series.
print("Shapes:", X.shape, y.shape) # Check alignment.
assert len(X) == len(y) # Check one target per example.
''',
'The hours column contains one missing value. Its representation becomes floating point because of that missing entry. Double brackets select a DataFrame; single brackets select a Series. Dropping rows from only X would misalign examples and targets.',
'Add a category column and print its unique values. Identify which column is the target and which columns would be available before the outcome occurs.',
'Assign df["group"] = ["A", "B", "A", "B"], then call df["group"].unique(). Keep score out of X.',
'Why keep the target out of the feature table?', 'Including it makes evaluation more realistic.', 'The true outcome would not be available when predicting a new case; including it leaks the answer.', 'KeyError: hours','Check df.columns for spelling, spaces, and capitalization before selecting a named column.')
L('Visualization and data inspection',
'Plots help reveal structure that a summary number can hide. A histogram shows how one variable is distributed. A scatter plot compares two numerical variables. A category count shows representation. Correlation describes a particular association; it does not establish causation, and a low linear correlation can hide a nonlinear pattern.',
'An average road speed does not tell you where the traffic jams are. A map of the journey reveals the pattern.',
'Inspect development data, then select plots based on the question. Label axes and units. Start with missingness and distributions, then examine relationships. Avoid making many modeling decisions after inspecting the final test outcomes. This example saves a plot rather than requiring an interactive display.',
'Histogram=A plot counting values within numerical intervals.|Scatter plot=A plot showing paired numerical measurements.|Outlier=An observation unusual relative to the data or context.|Correlation=A numerical measure of association, not proof of cause.',
'''import numpy as np # Load numerical tools.
import matplotlib.pyplot as plt # Load plotting tools.
x = np.array([1, 2, 3, 4, 5]) # Study time in hours.
y = np.array([15, 25, 32, 43, 49]) # Example scores.
fig, ax = plt.subplots() # Create a figure and axes.
ax.scatter(x, y) # Draw one point per example.
ax.set(xlabel="Study hours", ylabel="Score", title="Inspect before modeling") # Name the quantities.
fig.savefig("inspection.png", bbox_inches="tight") # Save a viewable plot.
print("Saved inspection.png; points:", len(x)) # Report the artifact and sample count.
print("Correlation:", round(np.corrcoef(x, y)[0, 1], 3)) # Summarize linear association.
assert len(ax.collections[0].get_offsets()) == 5 # Verify the plotted sample count.
plt.close(fig) # Release the figure resource.
''',
'Open the saved image or view the figure in a notebook. The positive association suggests scores rise with study hours in these five constructed examples. The sample is too small and artificial to support a claim about real students.',
'Plot a histogram and scatter plot for a real development dataset. Write one observation and one limitation for each.',
'Use ax.hist for one numeric column and ax.scatter for two. State sample size, units, missing-value handling, and whether the pattern could reflect confounding.',
'Does a strong correlation prove that changing one feature changes the target?', 'Yes; a value close to one proves causation.', 'No. Association can arise from confounding, selection, or other causes; the data collection design matters.', 'TypeError: no numeric data to plot','Inspect dtypes and convert only genuinely numeric columns; investigate nonnumeric entries first.')
L('Generalization, splitting, and leakage',
'A model should be evaluated on examples that did not teach it. Training examples fit parameters, validation examples support development decisions, and the final test set estimates performance after those decisions. A test set repeatedly used to choose settings effectively becomes another validation set.',
'Learning the answers to the exam is different from learning how to solve the questions.',
'Reserve the final test set first. Within development data, create training and validation partitions. Stratification approximately preserves label proportions but does not prevent the same person appearing in both partitions. Detect target-revealing features, duplicated entities, preprocessing learned on held-out rows, and information created after the event.',
'Generalization=Performance on relevant unseen examples.|Validation set=Examples used for development decisions.|Test set=Reserved examples used after the workflow is selected.|Leakage=Information entering model development that should not be available.',
CLASS+'''X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=7) # Split development data only.
model.fit(X_train, y_train) # Learn using the training partition.
print("Train, validation, test rows:", len(X_train), len(X_val), len(X_test)) # Account for every role.
print("Validation accuracy:", round(model.score(X_val, y_val), 3)) # Evaluate a development candidate.
assert len(X_train) + len(X_val) + len(X_test) == len(X) # No rows lost by the partitioning.
''',
'The three counts have different roles even though they come from one original table. The validation score can guide development. The test rows have been counted but not evaluated. A good validation result cannot detect all forms of leakage by itself.',
'Explain why using a discharge diagnosis to predict the admission diagnosis can be leakage. Identify the point in time when every feature becomes available.',
'A feature recorded after the prediction event can reveal information from the future. Write a prediction-time contract and remove unavailable features before evaluating.',
'When should the reserved final test set choose model settings?', 'After every training run.', 'It should not choose settings. Use development validation or cross-validation, then evaluate the selected workflow on the reserved test set.')
L('Transformers and pipelines',
'A transformer learns a rule for changing inputs and then applies that rule. A scaler learns averages and spreads; an imputer learns replacement values. A pipeline fits the steps in order and uses the learned rules consistently for future predictions. It does not fix an inappropriate split or target-revealing features.',
'An assembly line uses the same preparation stages for every new item.',
'Call fit only on the training partition. Call transform on validation and test inputs using the fitted transformer. A pipeline automates these steps. During prediction, preprocessing runs first and the final estimator predicts. fit_transform combines learning and transformation for training inputs; it is not the operation to use on a test set.',
'Transformer=A tool that changes the representation of input data.|Pipeline=An ordered preprocessing-and-estimation workflow.|fit_transform=Learn a transformation and apply it to the supplied data.|Training-only preprocessing=Learning transformation settings without validation or test rows.',
CLASS+'''model.fit(X_dev, y_dev) # Learn the scaler and classifier using development rows.
scaler = model.named_steps["standardscaler"] # Access a named fitted step.
print("Scaler learned features:", scaler.n_features_in_) # Check fitted input width.
print("First transformed row:", scaler.transform(X_dev[:1])[0, :3].round(3)) # Inspect three scaled values.
print("Prediction:", model.predict(X_dev[:1])) # Pass raw measurements to the pipeline.
assert scaler.n_features_in_ == X_dev.shape[1] # Check that preprocessing matches the inputs.
''',
'The pipeline accepts raw feature measurements. Negative scaled values generally mean below the training mean, not invalid measurements. Looking at a prediction on a development row illustrates the API; it is not an unbiased evaluation.',
'Replace the classifier with another estimator only after identifying whether that estimator needs scaling. Explain why the scaler must remain part of cross-validation.',
'Cross-validation must refit the scaler separately inside every training fold. Pre-scaling the whole development table lets validation rows influence preprocessing.',
'Which object should be sent to cross-validation?', 'Only the classifier after scaling all development data.', 'Send the whole pipeline so each validation fold is excluded from learning preprocessing.')
L('Introductory cross-validation',
'One validation partition can be easier or harder than another. Cross-validation repeats evaluation over several partitions. Each round fits a fresh estimator. It is not a model that keeps learning from one fold to the next. For this classification example, we use accuracy only; regression loss scorers arrive in Chapter 11.',
'A student sits several practice papers instead of being judged by one unusually easy paper.',
'Use development data only. StratifiedKFold keeps class proportions approximately similar across folds for classification. KFold is the general fold splitter, useful when stratification is not needed. Evaluate the pipeline and the baseline on the same folds. Inspect the average and individual scores. Fold standard deviation describes variability; it is not automatically a confidence interval.',
'Fold=A partition used for validation in one round.|Cross-validation=Repeated fitting and validation over partitions.|Stratification=Approximately preserving class proportions across partitions.|Score spread=Variation among validation scores, not a guaranteed uncertainty interval.',
CLASS+'''from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42) # Reproducible class-aware folds.
scores = cross_val_score(model, X_dev, y_dev, cv=cv, scoring="accuracy", error_score="raise") # Fit a fresh pipeline each round.
base = cross_val_score(DummyClassifier(strategy="most_frequent"), X_dev, y_dev, cv=cv, scoring="accuracy") # Use identical partitions.
print("Model folds:", scores.round(3)) # Inspect each validation score.
print("Mean and spread:", round(scores.mean(), 3), round(scores.std(), 3)) # Summarize scores.
print("Baseline mean:", round(base.mean(), 3)) # Compare with the reference rule.
assert scores.shape == (4,) # One score for each fold.
''',
'Each number in the model array is an accuracy on a different validation fold. The mean summarizes these rounds. The original model object is not left fitted by cross_val_score because cloned estimators are fitted internally. No final test score is used to choose a winner here.',
'Change to five folds. Report model and baseline means. Explain why a different number of folds changes both the training sizes and validation partitions.',
'Set n_splits=5 for both evaluations through the same splitter. Compare averages and individual scores without selecting the fold count to make the result look best.',
'Does cross-validation continue training the same fitted model across folds?', 'Yes; each fold adds more learned weights.', 'No. Each fold fits a fresh clone, preventing earlier validation examples from teaching later rounds indirectly.')
L('Linear regression',
'Linear regression predicts a number by multiplying each feature by a learned coefficient and adding an intercept. With one input, prediction = weight × input + intercept. A residual is the observed target minus the prediction. Least squares chooses coefficients that minimize squared residuals. Gradient descent is one way to optimize a loss, but sklearn LinearRegression uses least-squares solvers rather than a gradient-descent loop.',
'A straight ruler approximates a trend; its tilt is the weight and its vertical starting point is the intercept.',
'For X=[1,2,3], y=[3,5,7], begin with w=0 and b=0. Predictions are all zero and errors prediction−target are −3,−5,−7. For J=sum(error²)/(2m), the gradients are mean(error×X)=−34/3 and mean(error)=−5. At learning rate 0.1, one update gives w≈1.1333, b=0.5. Repeating updates can improve the line; the code below directly solves least squares and finds w=2, b=1.',
'Coefficient=A learned multiplier for one input feature.|Intercept=The prediction offset added to the weighted inputs.|Residual=Observed target minus predicted target.|Gradient=The local rate and direction of change of a loss with respect to a parameter.',
'''import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1], [2], [3]]) # Three one-feature examples.
y = np.array([3, 5, 7]) # Known numerical targets.
model = LinearRegression().fit(X, y) # Solve the least-squares problem.
pred = model.predict(X) # Compute fitted predictions.
print("Weight:", model.coef_.round(3)) # Multiplier per feature.
print("Intercept:", round(model.intercept_, 3)) # Offset.
print("Residuals:", (y - pred).round(6)) # Compare actual and predicted values.
print("At x=4:", model.predict([[4]]).round(3)) # Predict a new feature value.
assert np.allclose(pred, y) # This constructed relationship is exactly linear.
''',
'The coefficient 2 means one extra input unit increases the fitted prediction by two target units. The intercept 1 is the fitted value at zero input, which may be outside the observed range. Zero training residuals here reflect three perfectly constructed points; they do not demonstrate real-world generalization.',
'Compute the first gradient-descent update yourself using learning rate 0.01 instead of 0.1. Then add a second feature and inspect the number of coefficients.',
'The first update is w=0.113333..., b=0.05. With two feature columns the fitted coefficient array has two entries. Avoid interpreting coefficients causally.',
'Does LinearRegression.fit execute the hand-written gradient-descent loop shown in the explanation?', 'Yes; every regression estimator uses gradient descent.', 'No. LinearRegression solves least squares with its supported solvers; gradient descent explains a general optimization approach.')
L('Evaluating regression',
'A regression error must be interpreted in the target’s units. MAE averages absolute errors. MSE averages squared errors, giving large misses more influence. RMSE is the square root of MSE and returns to target units. R² compares squared error with a mean reference; it can be negative and is not a percentage accuracy.',
'A delivery estimate that misses by ten minutes should be described in minutes, not only as a model score.',
'For true values [10,20,30] and predictions [12,17,31], absolute errors are [2,3,1]. MAE=2; MSE=(4+9+1)/3=14/3; RMSE≈2.160. R²=1−14/200=0.93 for these examples. In cross-validation, neg_mean_absolute_error returns the negative of MAE because scorers use a higher-is-better convention. Negate it for reporting. Plot residuals against predictions to look for systematic patterns.',
'MAE=The mean absolute prediction error, in target units.|MSE=The mean squared prediction error, in squared target units.|RMSE=The square root of MSE, in target units.|R-squared=A comparison with a mean-reference squared error that can be negative.',
'''import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
y = np.array([10, 20, 30]) # Actual values.
pred = np.array([12, 17, 31]) # Predicted values.
print("MAE:", mean_absolute_error(y, pred)) # Average absolute miss.
print("MSE:", round(mean_squared_error(y, pred), 3)) # Average squared miss.
print("RMSE:", round(root_mean_squared_error(y, pred), 3)) # Restore target units.
print("R2:", round(r2_score(y, pred), 3)) # Compare squared errors to a reference.
assert mean_absolute_error(y, pred) == 2 # Verify the hand calculation.
''',
'All four metrics describe the same predictions in different ways. If target values were expressed in hundreds of thousands of dollars, multiply MAE or RMSE by 100,000 to report dollars. RMSE is not the average absolute miss. A baseline must be fitted on training targets, not constructed from held-out answers during model development.',
'Use cross_val_score with scoring="neg_mean_absolute_error" on a regression pipeline. Report positive fold errors and compare with DummyRegressor.',
'Fit both candidates within identical development folds. Set errors=-scores before averaging. Do not describe R² as the fraction of correctly predicted examples.',
'Is R²=0.9 the same as 90% of predictions being correct?', 'Yes; R² is regression accuracy.', 'No. It compares squared errors with a reference. Numerical prediction errors need units and an appropriate loss.')
L('Logistic regression and classification',
'Logistic regression is a classifier despite its name. In binary classification it maps a weighted input score into a value between zero and one using a sigmoid. A probability estimates an outcome under the model; it is not a guarantee. A decision threshold turns a probability into a label. Multiclass outputs contain a probability per class.',
'A weather forecast can assign a chance of rain before a separate decision about carrying an umbrella.',
'Read classes_ before interpreting probability columns. For binary classification, the default probability-based decision boundary is generally 0.5; choosing a different threshold changes false positives and false negatives. For a probability of 0.8, repeated comparable predictions should be positive about 80% of the time if well calibrated. Chapter 27 examines calibration.',
'Class=A category a classifier can predict.|Probability=A value from zero to one quantifying modeled likelihood.|Threshold=A cutoff used to turn a score into a decision.|Class order=The order of labels corresponding to probability columns.',
CLASS+'''model.fit(X_dev, y_dev) # Fit the complete classification workflow.
print("Classes:", model.classes_) # Inspect label order.
probabilities = model.predict_proba(X_test[:3]) # Three rows, one column per class.
print("Probabilities:", probabilities.round(3)) # Inspect the numerical estimates.
print("Labels:", model.predict(X_test[:3])) # Inspect the resulting decisions.
print("Row sums:", probabilities.sum(axis=1).round(3)) # Each row is a class distribution.
assert probabilities.shape == (3, 2) # Binary classification has two columns here.
''',
'Each row contains two probabilities whose sum is approximately one. The larger probability determines the default predicted class in this binary example. Whether the numerical values are trustworthy requires evaluation beyond accuracy.',
'Explain which probability column represents label 1 using classes_. Why does a model probability of 0.8 not guarantee the outcome?',
'Find the index where classes_ equals 1. A probability expresses uncertainty across comparable cases, not certainty about one individual.',
'What determines the label associated with a predict_proba column?', 'The column is always positive class first.', 'The fitted classes_ attribute defines the label order; never assume a fixed semantic meaning.')
L('Evaluating classification',
'A confusion matrix counts predictions by actual and predicted class. Choose which class is positive before computing precision and recall. For binary labels, true positives are correctly identified positives; false negatives are missed positives. Precision asks how many predicted positives were real. Recall asks how many actual positives were found.',
'A fishing net can catch many fish but also debris; recall describes fish caught and precision describes the purity of the catch.',
'With TP=8, FP=2, FN=4, TN=6, accuracy=(8+6)/20=0.70, precision=8/10=0.80, recall=8/12≈0.667, and F1=2TP/(2TP+FP+FN)=16/22≈0.727. For multiple classes, macro averages give each class equal weight; weighted averages use support; micro aggregates counts. ConfusionMatrixDisplay makes counts easier to inspect. Report the chosen label and averaging method.',
'Precision=The fraction of predicted positives that are actually positive.|Recall=The fraction of actual positives that were found.|F1=The harmonic mean of precision and recall.|Support=The number of actual examples of a class.',
'''from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
y = [1] * 12 + [0] * 8 # Twelve actual positives and eight actual negatives.
pred = [1] * 8 + [0] * 4 + [1] * 2 + [0] * 6 # Eight true positives, four misses, two false alarms.
print("Matrix (rows actual):", confusion_matrix(y, pred).tolist()) # Default label order is 0, then 1.
print("Accuracy:", round(accuracy_score(y, pred), 3)) # Overall fraction correct.
print("Precision:", round(precision_score(y, pred), 3)) # Purity of positive predictions.
print("Recall:", round(recall_score(y, pred), 3)) # Coverage of actual positives.
print("F1:", round(f1_score(y, pred), 3)) # Balance both quantities.
assert confusion_matrix(y, pred).sum() == len(y) # Every example is counted once.
''',
'The first matrix row corresponds to actual label 0. Default ordering gives [[TN, FP], [FN, TP]]. A model can have high overall accuracy while missing most rare positives. Neither precision nor recall alone describes every type of error.',
'Create a ConfusionMatrixDisplay for a digits classifier. Name the two most frequently confused distinct labels and explain why diagonal counts are excluded.',
'Use ConfusionMatrixDisplay.from_predictions. Set diagonal matrix entries to zero in a copy before finding the largest off-diagonal count. Diagonal entries are correct predictions.',
'What does recall measure?', 'The fraction of predicted positives that were correct.', 'Recall is TP/(TP+FN), the fraction of actual positives detected. The alternative describes precision.')
L('Missing and invalid values',
'Missingness has causes. A blank age can mean not collected, declined, or lost during processing. Filling a value does not recover the true measurement; it provides a consistent representation for a model. Invalid values such as impossible ages require domain checks before imputation. Indicators can preserve whether a value was originally absent.',
'A substitute ingredient lets cooking continue but should not be mistaken for the original ingredient.',
'Inspect missingness by feature and subgroup. Fit an imputer on training data only, ideally inside a pipeline. SimpleImputer can use a mean, median, most-frequent value, or constant. KNNImputer borrows values from nearby rows and needs more care with feature scales and computation. Check the policy for columns that are entirely missing.',
'Imputation=Replacing absent values using a specified rule.|Median=The middle value after sorting observations.|Missing indicator=A feature recording whether a value was absent.|Invalid value=A present entry that violates the meaning or allowed range of a feature.',
'''import numpy as np
from sklearn.impute import SimpleImputer
train = np.array([[1, 10], [3, np.nan], [5, 30]]) # Training examples with one missing value.
new = np.array([[np.nan, 50]]) # A future example.
imputer = SimpleImputer(strategy="median", add_indicator=True) # Learn medians and missingness markers.
imputer.fit(train) # Learn from training examples only.
print("Medians:", imputer.statistics_) # Replacement value per original feature.
print("Transformed new row:", imputer.transform(new)) # Apply learned values without refitting.
assert np.isfinite(imputer.transform(new)).all() # Check finite numeric output.
''',
'The learned medians are 3 and 20. With the default indicator behavior, an indicator is added for the second column because it was missing during fitting; no indicator is added for the first column even though it is missing in the new row. This is why understanding output columns matters.',
'Compare mean and median imputation after adding an extreme training value. Explain why neither method should fit on the final test set.',
'The mean responds strongly to an extreme value; the median often changes less. Both replacement rules learn from data and must exclude held-out information.',
'Does filling a missing value recover the original unknown measurement?', 'Yes; imputation reveals the true value.', 'No. It creates an estimate or representation using a chosen rule, whose effect must be evaluated.')
L('Categorical encoding',
'An unordered category such as browser type should not be given an artificial numeric distance. One-hot encoding creates category indicator columns. Ordinal encoding is appropriate when order is meaningful or when a model’s documented behavior supports the representation. Target encoding uses target information, so unsafe fitting can leak outcomes.',
'Assigning jersey numbers does not mean player 20 is twice player 10.',
'Fit the category vocabulary on training rows. Decide how to handle categories that were not observed. OneHotEncoder(handle_unknown="ignore") uses all zeros for unknown categories in the corresponding feature block. TargetEncoder.fit_transform uses internal cross-fitting for training representations; fit(...).transform(...) has different leakage implications. Supervised encoders belong inside model-selection folds.',
'Nominal category=A category with no inherent ranking.|Ordinal category=A category with a meaningful order.|One-hot encoding=Representing categories with separate indicator features.|Cross-fitting=Creating training representations using models fitted on other partitions.',
'''import pandas as pd
from sklearn.preprocessing import OneHotEncoder
train = pd.DataFrame({"city": ["Akure", "Lagos", "Akure"]}) # Learn a category vocabulary.
new = pd.DataFrame({"city": ["Abuja"]}) # A category not present in training.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False) # Request dense output for this tiny example.
encoder.fit(train) # Learn only training categories.
print("Columns:", encoder.get_feature_names_out().tolist()) # Inspect generated feature names.
print("Known:", encoder.transform(train.iloc[:1]).tolist()) # Encode an observed category.
print("Unknown:", encoder.transform(new).tolist()) # Apply the specified unknown-category policy.
assert encoder.transform(new).shape[1] == 2 # New categories must not change feature width.
''',
'The known city activates one indicator. The unknown city activates neither known-city indicator. That prevents a crash, but does not mean the model has learned a special Abuja effect. Dense output is fine here; a large category vocabulary often needs sparse storage.',
'Add an ordered rating column with low, medium, high. Specify its order explicitly with OrdinalEncoder and explain why alphabetical order is not enough.',
'Use categories=[["low", "medium", "high"]]. The representation should match the real semantic order, not an arbitrary sorting convention.',
'Why does target encoding need special precautions?', 'It never uses target values.', 'It uses target information to create features, so training representations and evaluation folds need leakage-aware fitting.')
L('Scaling and feature engineering',
'Scaling changes numerical representation. StandardScaler subtracts training means and divides by training standard deviations. MinMaxScaler uses the training range. RobustScaler uses robust summaries such as a median and interquartile range. Scaling does not require or force a feature to become normally distributed.',
'Converting metres and kilometres to a common scale keeps one measurement from dominating a distance calculation.',
'Use scaling when an algorithm is sensitive to distances or coefficient penalties. Tree splits often need no scaling. Feature engineering constructs useful inputs such as age at prediction time or a ratio. Every ingredient must be available at that time. Polynomial features can expand a feature space rapidly and require regularization and cross-validation.',
'Standardization=Subtracting a training mean and dividing by a training standard deviation.|Robust scaling=Scaling based on summaries less sensitive to extremes.|Interaction=A feature representing a joint relationship between inputs.|Feature engineering=Constructing inputs from available raw information.',
'''import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
X = np.array([[1, 100], [2, 200], [3, 300]]) # Features with different units.
scaler = StandardScaler().fit(X) # Learn per-feature means and scales.
scaled = scaler.transform(X) # Apply the learned numerical conversion.
print("Means:", scaler.mean_) # Inspect original-unit averages.
print("Scaled means:", scaled.mean(axis=0).round(6)) # Training means are near zero.
poly = PolynomialFeatures(degree=2, include_bias=False) # Add squares and interactions.
print("Expanded names:", poly.fit(X).get_feature_names_out().tolist()) # Inspect added inputs.
assert np.allclose(scaled.mean(axis=0), 0) # Check this training transformation.
''',
'Zero scaled training means are expected by construction. Future data need not have zero means under the same fitted scaler. Polynomial expansion creates x0, x1, squares, and an interaction; more columns do not automatically improve generalization.',
'Compare a scaled and unscaled nearest-neighbor pipeline on identical folds. Keep feature engineering inside each evaluated pipeline.',
'Use make_pipeline(StandardScaler(), KNeighborsClassifier()) and an unscaled comparator with the same splitter. Explain differences using distance and feature units.',
'Does StandardScaler guarantee a normal distribution?', 'Yes; it makes every histogram Gaussian.', 'No. It changes mean and scale, not the underlying distribution shape.')
L('Mixed-data pipelines',
'Real tables often mix numbers and categories. ColumnTransformer sends selected columns through different preprocessing branches, then joins their outputs. Pipeline connects that prepared table to a predictor. The resulting object accepts raw records with the same input contract.',
'A sorting station sends glass and paper through different processing lines before assembling the final package.',
'Use a numeric branch for imputation and scaling and a categorical branch for imputation and encoding. Inspect generated feature names. set_output(transform="pandas") can help retain names, but pandas output is incompatible with sparse output from OneHotEncoder unless a compatible dense configuration is selected. Use nested names such as model__C for tuning a named pipeline step.',
'ColumnTransformer=A tool applying different transformations to selected columns.|Branch=One preprocessing path for a subset of features.|Feature name=A label identifying an original or generated input.|Nested parameter=A setting addressed through its containing step using double underscores.',
'''import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
X = pd.DataFrame({"size": [1, 2, 3, 4], "city": ["A", "B", "A", "B"]}) # Raw mixed inputs.
y = [10, 22, 30, 42] # Numerical targets.
prep = ColumnTransformer([("num", SimpleImputer(), ["size"]), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"])]) # Define both branches.
prep.set_output(transform="pandas") # Preserve generated names in a table.
model = make_pipeline(prep, LinearRegression()).fit(X, y) # Fit all steps together.
print("Features:", prep.get_feature_names_out().tolist()) # See the joined representation.
print("New prediction:", model.predict(pd.DataFrame({"size": [5], "city": ["C"]})).round(2)) # Accept an unknown category.
assert len(model.predict(X)) == 4 # Predict once per raw input row.
''',
'The output has one numerical feature and one indicator per training category. The new record uses a previously unseen city; the encoder follows its ignore policy. A valid prediction is not evidence the model learned how the unseen city behaves.',
'Build separate named steps prep and model with Pipeline. Write the parameter name for the classifier’s regularization setting C.',
'For Pipeline([("prep", prep), ("model", LogisticRegression())]), use model__C. Names follow the exact step names you chose.',
'Why use separate numeric and categorical branches?', 'Every model directly understands raw strings.', 'Different feature types need different preprocessing rules before they can be combined into one model input.')
L('Transforming regression targets',
'Target transformation changes the quantity a regressor learns. For a strongly skewed nonnegative target, log1p can reduce the influence of very large values. A wrapper can reverse that transformation when predicting. Changing the training scale changes the objective, so improvements must be judged in meaningful original units.',
'Compressing a large map makes it easier to handle, but distances must be converted back before reporting a journey.',
'Use TransformedTargetRegressor with log1p and expm1 as an inverse pair. Ensure the target domain permits the transformation. The returned predict values are already inverse-transformed. An inverse of a predicted mean on the log scale is not automatically the mean on the original scale; interpret the objective carefully.',
'Target transformation=A mapping applied to y before fitting.|Inverse transformation=A mapping returning predictions to the original scale.|log1p=The natural logarithm of one plus a value.|Original-unit evaluation=Calculating prediction errors after returning to the target’s real units.',
'''import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
X = np.array([[1], [2], [3], [4]]) # One input per example.
y = np.expm1(1 + 0.5 * X.ravel()) # Construct a positive exponential relationship.
model = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1) # Pair forward and reverse maps.
model.fit(X, y) # Fit the transformed training target.
pred = model.predict(X) # Automatically return predictions to original units.
print("Predictions:", pred.round(2)) # Inspect original-scale values.
print("Original-unit MAE:", round(mean_absolute_error(y, pred), 6)) # Evaluate on the reported scale.
assert np.allclose(pred, y) # This constructed relationship fits exactly.
''',
'The target was deliberately constructed to be linear after log1p, so near-zero training error is expected. Real targets usually retain noise and mismatch. A fair comparison evaluates transformed and untransformed candidates on the same held-out partitions.',
'Explain why applying expm1 again to model.predict would be incorrect. Identify a target where log1p is not directly valid.',
'predict already reverses the transformation. Values at or below −1 are outside the real-valued log1p domain; choose a domain-appropriate transformation.',
'What scale does TransformedTargetRegressor.predict return in this example?', 'The logarithmic training scale.', 'It returns the original target scale by applying inverse_func to the internal regressor predictions.')
L('Complexity and regularization',
'A flexible model can fit accidental patterns in training data. Underfitting means the model misses important structure even in training; overfitting means training success fails to transfer to held-out examples. Bias–variance describes a tradeoff between systematic mismatch and sensitivity to the particular training sample. It is a useful diagnostic idea, not a reason every model must occupy a fixed complexity ranking.',
'A curve through every practice point may twist wildly between them.',
'Compare training and validation errors. Ridge penalizes squared coefficients, Lasso penalizes absolute coefficients and can set some to zero, and Elastic Net combines both. Alpha controls penalty strength. Scale inputs when coefficient penalties should treat feature units comparably. Polynomial features plus regularization are a controlled way to explore nonlinear structure.',
'Underfitting=Missing important structure with an insufficiently suitable model.|Overfitting=Learning training-specific patterns that do not transfer well.|Regularization=A constraint or penalty that limits fitted complexity.|Alpha=The penalty-strength parameter in estimators such as Ridge.',
REG+'''from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
for alpha in [0.1, 10, 100]: # Compare a small prespecified set of penalties.
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)) # Scale and penalize coefficients.
    errors = -cross_val_score(model, X_dev, y_dev, cv=4, scoring="neg_mean_absolute_error") # Evaluate without final-test selection.
    print("alpha, mean MAE:", alpha, round(errors.mean(), 2)) # Lower positive error is better.
assert len(errors) == 4 # Four held-out scores per candidate.
''',
'Alpha changes the penalty, not the learning rate. Stronger penalties can help or hurt depending on the data. Choose settings using development evaluation; the final test must not pick the best alpha.',
'Add polynomial features to a noisy one-feature dataset. Compare training and validation errors for degree 1 and degree 10, with and without Ridge.',
'Put PolynomialFeatures and StandardScaler inside each evaluated pipeline. More training fit with worse validation is evidence consistent with overfitting.',
'Is a larger regularization alpha always better?', 'Yes; stronger regularization always improves accuracy.', 'No. Too much penalty can underfit. Compare candidates using suitable development evaluation.')
L('Nearest neighbors and decision trees',
'Nearest neighbors predicts from similar training examples. Similarity depends on the distance rule and feature scales. A decision tree recursively splits data using feature thresholds. Trees can model nonlinear relationships but deep trees may memorize small training regions. Both models have settings that control sensitivity to noise.',
'Neighbors ask nearby classmates; a tree follows a sequence of yes-or-no questions.',
'For neighbors, compare several values of n_neighbors in a scaled pipeline. For trees, inspect max_depth and min_samples_leaf. A minimum leaf size prevents tiny groups from defining predictions. Fit choices inside cross-validation; never infer a model’s usefulness only from its training score.',
'Neighbor=A training example close under a chosen distance rule.|Leaf=A terminal tree region producing a prediction.|Depth=The length of a decision path in a tree.|Minimum leaf size=The smallest allowed number of training examples in a leaf.',
CLASS+'''from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(3, shuffle=True, random_state=42) # Shared validation partitions.
models = {"neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(5)), "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)} # Define candidates.
for name, candidate in models.items(): # Evaluate each candidate fairly.
    scores = cross_val_score(candidate, X_dev, y_dev, cv=cv, scoring="accuracy") # Refit within each fold.
    print(name, round(scores.mean(), 3)) # Compare validation accuracy.
assert len(scores) == 3 # Check fold count.
''',
'A depth limit restricts a tree’s paths. Neighbor count controls how local the average or vote is. Scaling is important for the neighbor distance here, while a tree’s threshold splits usually do not need it.',
'Change only the tree depth, then only the neighbor count. Explain why changing several things at once makes the result harder to interpret.',
'Prespecify a small comparison and retain the same splitter. One-factor changes isolate which decision caused a performance difference.',
'Which model is directly sensitive to feature units in its distance computation?', 'A threshold tree always needs equal feature units.', 'Nearest neighbors uses distances, so a large-scale feature can dominate unless the representation is appropriate.')
L('Random forests and Extra Trees',
'A random forest combines many randomized trees. Bagging fits trees on resampled training rows; random feature selection reduces similarity between trees. Averaging or voting can reduce instability compared with one tree. Extra Trees adds more randomization to split selection. More trees can improve stability while increasing storage and prediction cost.',
'A panel of partly independent opinions can be more stable than one person’s judgment.',
'Compare candidates on the same folds. Consider n_estimators, max_features, min_samples_leaf, and depth. Built-in impurity importances can favor features with many potential split points and do not prove causality. Use held-out permutation importance later. Keep the search budget small enough to interpret and reproduce.',
'Bagging=Combining models fitted on resampled training examples.|Ensemble=A prediction system combining several fitted models.|Random forest=A collection of trees with randomized training and feature selection.|Impurity importance=A split-based importance summary that can be biased.',
CLASS+'''from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(3, shuffle=True, random_state=42) # Keep evaluation consistent.
for estimator in [RandomForestClassifier, ExtraTreesClassifier]: # Compare two tree ensembles.
    candidate = estimator(n_estimators=40, min_samples_leaf=2, random_state=42, n_jobs=1) # Bound computation and seed randomness.
    scores = cross_val_score(candidate, X_dev, y_dev, cv=cv, scoring="accuracy") # Evaluate the unfitted candidate.
    print(estimator.__name__, round(scores.mean(), 3)) # Report comparable scores.
assert len(scores) == 3 # One result per fold.
''',
'The class name identifies the algorithm, not a performance guarantee. n_jobs=1 avoids nested parallelism in this small example. A tiny score difference may be less meaningful than runtime or memory differences.',
'Compare 20 and 100 trees while recording fit time and serialized model size. State which tradeoff matters for your application.',
'Use time.perf_counter for fitting and joblib with consistent compression for size. Compare the same data and hardware, and report timing as environment-dependent.',
'Does high impurity importance prove a feature causes the target?', 'Yes; trees establish causal direction.', 'No. Importance describes model use under particular data and settings, and can be biased or shared among correlated features.')
L('Gradient boosting',
'Boosting adds models sequentially to improve an objective. In gradient boosting, each new learner targets directions that reduce the current loss. Learning rate controls how strongly each learner contributes. Histogram-based estimators group feature values into bins to make training more efficient. Early stopping uses validation evidence to stop adding iterations.',
'A team revises a draft in rounds, with each round addressing weaknesses in the current version.',
'Balance learning rate with the iteration budget. Compare validation curves rather than assuming more iterations help. Keep the early-stopping validation data separate from the final test set. When cross-validating, internal validation must remain within the current training fold. Missing-value and categorical support depend on the estimator and representation.',
'Boosting=Sequentially combining learners to reduce an objective.|Learning rate=The strength of each boosting update.|Histogram bin=A grouped range of feature values used to speed split search.|Early stopping=Ending training when an appropriate validation rule stops improving.',
CLASS+'''from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
candidate = HistGradientBoostingClassifier(max_iter=40, learning_rate=0.1, max_leaf_nodes=15, early_stopping=True, random_state=42) # Define bounded boosting.
cv = StratifiedKFold(3, shuffle=True, random_state=42) # Evaluate on development data.
scores = cross_val_score(candidate, X_dev, y_dev, cv=cv, scoring="accuracy") # Internal early stopping sees training-fold data only.
print("Fold accuracies:", scores.round(3)) # Inspect spread before choosing.
print("Mean:", round(scores.mean(), 3)) # Summarize evaluation.
assert len(scores) == 3 # Verify fold count.
''',
'max_iter is an upper limit, not a promise that all iterations run. Early stopping can end fitting earlier. Lower learning rates often need more iterations, so comparing only one setting of each can be misleading.',
'Compare two learning rates with a suitable iteration budget. Explain why the final test set cannot be used as the early-stopping set.',
'Early stopping makes training decisions from validation feedback. Using the final test for that purpose consumes its independence.',
'Why pair learning rate with iteration budget?', 'They control unrelated file formats.', 'Smaller updates often require more iterations; a fair comparison considers both training strength and budget.')
L('Choosing the right split',
'The evaluation split should imitate the intended deployment. Records from one person or device can share distinctive patterns. Randomly splitting their rows lets the model recognize an entity already seen. Future prediction requires a split that respects time. Stratification maintains proportions, not independence.',
'Practising on one person’s handwriting and testing on their other pages does not measure performance on a new writer.',
'Use GroupKFold when groups must not overlap. For ordered observations, TimeSeriesSplit trains on earlier rows and validates on later rows. Sort by time first and consider a gap when feature windows or label horizons overlap. Neither splitter automatically fixes features created using future values.',
'Group=An entity whose related rows should stay together.|Temporal split=A split respecting observation order in time.|Gap=Rows excluded between training and validation boundaries.|Deployment match=An evaluation design resembling the actual prediction situation.',
'''import numpy as np
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
X = np.arange(24).reshape(12, 2) # Twelve ordered examples.
y = np.arange(12) # Matching targets.
groups = np.repeat([0, 1, 2, 3], 3) # Three rows from each entity.
for train, val in GroupKFold(4).split(X, y, groups): # Hold out whole entities.
    assert not set(groups[train]) & set(groups[val]) # Check no shared entity.
print("Group folds: no overlap") # Report the verified condition.
for train, val in TimeSeriesSplit(n_splits=3, gap=1).split(X): # Keep validation later than training.
    print("Last train / first validation:", int(train[-1]), int(val[0])) # Inspect boundaries.
    assert train[-1] < val[0] # Check temporal order.
''',
'Group disjointness is explicit in the first assertion. The time indices show expanding training windows and later validation windows. A one-row gap is illustrative; the correct gap depends on information availability and forecast horizon.',
'Identify whether a device-failure task should split by row, device, time, or a combination. State what deployment population you want to estimate.',
'For unseen devices, keep devices disjoint. For future readings on known devices, preserve time. Some applications need a custom combination of both constraints.',
'Does stratification prevent the same patient appearing in train and validation?', 'Yes; stratification separates identities.', 'No. Stratification preserves class proportions. Entity separation requires group-aware splitting.')
L('Hyperparameter search',
'Hyperparameters are settings chosen before fitting, unlike coefficients learned from data. Search tools evaluate candidate settings using cross-validation. Grid search enumerates a specified product of choices; randomized search samples a fixed number of candidates. A large search can overfit development evidence, so tuning is not a substitute for a reserved final test.',
'Trying many recipes is useful only if every recipe is judged by the same tasting procedure.',
'Put preprocessing inside the searched estimator. Nested parameter names follow step names separated by double underscores. Read mean_test_score, std_test_score, params, and fit times in cv_results_. With refit=True, the selected candidate is fitted again on all supplied development data. Search fits and the final refit are separate costs.',
'Hyperparameter=A model setting supplied before learning fitted parameters.|Grid search=Evaluation of every combination in a specified parameter grid.|Randomized search=Evaluation of a sampled set of parameter combinations.|Refit=Fitting the selected candidate on the full development input after search.',
CLASS+'''from sklearn.model_selection import GridSearchCV, StratifiedKFold
cv = StratifiedKFold(3, shuffle=True, random_state=42) # Share reproducible evaluation partitions.
search = GridSearchCV(model, {"logisticregression__C": [0.1, 1, 10]}, cv=cv, scoring="accuracy", refit=True, error_score="raise") # Search the pipeline setting.
search.fit(X_dev, y_dev) # Nine fold fits plus one selected refit.
print("Best settings:", search.best_params_) # Inspect the chosen hyperparameters.
print("Mean CV accuracy:", round(search.best_score_, 3)) # Development selection evidence.
print("Candidates:", len(search.cv_results_["params"])) # Count tried configurations.
assert len(search.cv_results_["params"]) == 3 # Verify the search space.
''',
'best_score_ is the selected cross-validation score, not a final-test result. The leading logisticregression name is generated by make_pipeline. If you rename the step, change the parameter prefix accordingly.',
'Count fits for a 20-candidate randomized search with five folds and refit=True. Name one way to reduce computation without using the test set.',
'There are 100 cross-validation fits plus one refit. Reduce candidate count, use a justified subset, or bound model complexity while retaining a suitable evaluation design.',
'What does best_score_ represent?', 'An unbiased performance guarantee on future populations.', 'It is the winning development cross-validation score after selection, not the reserved final-test result.')
L('Custom scorers and sample weights',
'A training objective and an evaluation metric serve different roles. Sample weights can change which training examples influence fitting, while evaluation weights change which examples contribute to reported performance. A custom scorer wraps a metric for model-selection tools. These operations must reflect the intended population and costs.',
'A teacher can emphasize certain practice questions during learning and separately assign marks during grading.',
'make_scorer tells search how to compare results. Set greater_is_better=False for a loss so returned scores are negated. Some scorers need probabilities or decision scores rather than labels. Passing sample_weight through nested estimators depends on supported parameters and metadata routing. Verify fit and scoring requests separately; weights are not automatically applied everywhere.',
'Scorer=A callable following the evaluation interface used by model-selection tools.|Sample weight=A value controlling one example’s contribution.|Cost-sensitive evaluation=Scoring predictions according to meaningful unequal error costs.|Metadata routing=Explicit rules for forwarding auxiliary data to compatible components.',
'''import numpy as np
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression
X = np.array([[1], [2], [3], [4]]) # A tiny regression input.
y = np.array([2, 4, 6, 20]) # One influential unusual target.
w = np.array([1, 1, 1, 0.1]) # Downweight the unusual row for this demonstration.
model = LinearRegression().fit(X, y, sample_weight=w) # Apply training weights directly.
scorer = make_scorer(mean_absolute_error, greater_is_better=False) # Negate a lower-is-better loss.
print("Weighted fit predictions:", model.predict(X).round(2)) # See the effect on fitting.
print("Negative MAE:", round(scorer(model, X, y), 3)) # This evaluation is unweighted.
print("Weighted MAE:", round(mean_absolute_error(y, model.predict(X), sample_weight=w), 3)) # Explicitly weight evaluation.
assert np.isfinite(scorer(model, X, y)) # Check a valid numeric score.
''',
'The fit weights changed the line. The custom scorer call is unweighted because no weights were supplied to that evaluation. The last metric explicitly applies evaluation weights. A negative loss score does not mean negative prediction errors.',
'Explain a legitimate reason to weight evaluation separately from training. Inspect metadata-routing examples in the versioned documentation before forwarding weights through a search.',
'Sampling weights may estimate a deployment population while training weights may optimize a cost-aware objective. In routed workflows configure compatible fit and score requests explicitly and verify each recipient.',
'Are weights passed to fit automatically applied to every evaluation?', 'Yes; sample weights affect every later score automatically.', 'No. Training and evaluation weighting are separate decisions and must be passed to the appropriate recipients.')
L('Imbalanced classification',
'When positives are rare, predicting every case as negative can have impressive accuracy and no useful recall. Class weights adjust fitting emphasis. A decision threshold changes which scores become positive predictions. These are different interventions. Evaluate at the natural deployment prevalence unless a clearly stated weighting scheme says otherwise.',
'A smoke alarm that never rings has few false alarms but is useless during a fire.',
'Report class prevalence and compare with a baseline. Precision–recall curves emphasize positive predictions; ROC-AUC measures ranking using false-positive and true-positive rates. ROC-AUC can still be valid for rare events, but a low false-positive rate may translate to many false alarms among numerous negatives. Average precision summarizes a precision–recall tradeoff and depends on prevalence. Select thresholds using development data only.',
'Prevalence=The fraction of examples belonging to a class.|Class weight=A fitting adjustment based on class membership.|Average precision=A summary of precision across recall changes.|Decision threshold=A cutoff converting scores or probabilities into class decisions.',
'''from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
X, y = make_classification(n_samples=600, n_features=8, weights=[0.95, 0.05], random_state=42) # Construct a rare-positive task.
Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, random_state=42) # Hold out development validation rows.
model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xt, yt) # Reweight fitting contributions.
p = model.predict_proba(Xv)[:, 1] # Read label-1 probabilities in this known binary task.
print("Validation prevalence:", round(yv.mean(), 3)) # Establish class frequency.
print("Average precision:", round(average_precision_score(yv, p), 3)) # Assess positive ranking.
for threshold in [0.3, 0.5, 0.7]: # Explore development-only decision cutoffs.
    pred = p >= threshold # Convert probabilities to decisions.
    print(threshold, round(precision_score(yv, pred, zero_division=0), 3), round(recall_score(yv, pred), 3)) # Inspect the tradeoff.
assert len(p) == len(yv) # One score per validation example.
''',
'Changing the threshold does not refit the model. Lower thresholds commonly increase recall and false positives, but report the actual observed tradeoff. Class weighting does not guarantee calibrated probabilities.',
'Choose a threshold based on a stated minimum recall or error cost using development predictions. Keep a separate final test set for the chosen rule.',
'Write the objective first, then compare validation thresholds. Do not repeatedly adjust a rule after observing final-test results.',
'Why can a high ROC-AUC coexist with too many false alarms?', 'ROC-AUC directly guarantees high precision.', 'Many negatives can produce many false positives even at a low false-positive rate; inspect precision and operational counts.')
L('Calibration and performance diagnosis',
'A classifier can rank cases well but assign misleading probabilities. Calibration checks whether predicted probabilities match observed frequencies for comparable predictions. Learning curves compare performance at different training sizes. Validation curves compare model settings. Diagnose a problem before prescribing more data or a more complex model.',
'A reliable fuel gauge should say half-full when the tank is roughly half-full, not merely rank tanks from emptier to fuller.',
'Use development partitions for calibration and diagnostics. CalibratedClassifierCV fits calibration without allowing calibration examples to directly train the corresponding base predictor. Inspect reliability curves and probability losses. Examine subgroup error counts and sample sizes; a small subgroup score is noisy. Explain possible harms and measurement limitations without treating group differences as causal findings.',
'Calibration=Agreement between predicted probabilities and observed outcome frequencies.|Learning curve=Performance evaluated at different training-set sizes.|Validation curve=Performance evaluated over a model setting.|Subgroup analysis=Examining performance for meaningful subsets with sufficient context.',
CLASS+'''from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
Xt, Xv, yt, yv = train_test_split(X_dev, y_dev, stratify=y_dev, random_state=7) # Reserve development validation examples.
calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3) # Calibrate through internal training folds.
calibrated.fit(Xt, yt) # Fit without final-test information.
p = calibrated.predict_proba(Xv)[:, 1] # Probabilities for class 1.
print("Validation Brier loss:", round(brier_score_loss(yv, p), 4)) # Evaluate squared probability error.
print("Probability range:", round(p.min(), 3), round(p.max(), 3)) # Inspect valid range.
assert ((p >= 0) & (p <= 1)).all() # Probabilities must be bounded.
''',
'Brier loss is lower-is-better; its value depends on the task and prevalence. Bounds from zero to one are necessary but do not prove calibration. Compare calibrated and uncalibrated workflows on the same held-out development examples.',
'Plot a calibration curve and a learning curve. Write a hypothesis about the main limitation, then name evidence that could disprove it.',
'Use CalibrationDisplay.from_estimator and learning_curve. Avoid diagnosing overfitting from one score alone; inspect training/validation patterns and variance.',
'Does a probability between zero and one prove calibration?', 'Yes; any bounded output is calibrated.', 'No. Calibration requires agreement with observed outcome frequencies on appropriate held-out examples.')
L('Text classification',
'Text models need a numerical representation. CountVectorizer counts token occurrences; TfidfVectorizer also downweights terms common across training documents. The vocabulary and document-frequency statistics must be fitted only on training text. Sparse matrices avoid storing the many zeros in a large text vocabulary.',
'A document becomes a shopping receipt recording which words appeared and how often.',
'Build a text pipeline so vocabulary learning is repeated inside each training fold. Compare logistic regression, LinearSVC, and MultinomialNB using identical splits. LinearSVC supplies decision_function scores rather than predict_proba. In newsgroup datasets, headers, footers, and quoted text can provide shortcuts; choose removal settings according to the intended task.',
'Token=A text unit such as a word produced by a tokenizer.|Vocabulary=The set of learned text features.|TF-IDF=A weighting based on term frequency and training-document frequency.|Sparse matrix=A representation storing mostly nonzero entries.',
'''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
texts = ["invoice payment due", "pay invoice today", "football team wins", "team scores goal", "payment receipt", "football match goal"] # Small labeled documents.
y = [0, 0, 1, 1, 0, 1] # Two topic labels.
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC(random_state=42)) # Learn words and a linear boundary.
model.fit(texts, y) # Fit vocabulary only on these training documents.
print("Predictions:", model.predict(["invoice receipt", "team goal"]).tolist()) # Predict new texts.
print("Feature count:", len(model.named_steps["tfidfvectorizer"].get_feature_names_out())) # Count learned tokens and bigrams.
assert len(model.predict(["new message"])) == 1 # Unknown words must not change output shape.
''',
'Labels 0 and 1 are defined by this toy dataset, not intrinsic meanings in the model. Bigram features represent two-token sequences. A tiny training set cannot support conclusions about production message classification.',
'Compare the three classifiers on a real text dataset with a held-out split. Explain why the vectorizer belongs inside every evaluated pipeline.',
'Fit the vectorizer separately within each fold. If using LinearSVC, use decision_function for ranking metrics that accept scores rather than assuming predict_proba exists.',
'Does LinearSVC provide predict_proba by default?', 'Yes; every classifier has predict_proba.', 'No. It provides decision_function scores; use a compatible scoring method or a separately evaluated calibration workflow.')
L('Clustering',
'Clustering searches for groups without target labels teaching the grouping. K-means seeks compact groups around centroids; DBSCAN groups dense regions and can mark noise; hierarchical clustering builds nested groups. Different methods define similarity differently, so clusters are hypotheses to investigate rather than discovered truths.',
'Sorting objects by color gives different piles from sorting them by shape.',
'Scale inputs when their units would dominate distance. Compare cluster stability and usefulness, not only a silhouette score. Silhouette compares within-cluster closeness with separation from other clusters and requires a valid number of clusters. Do not use an unavailable target to tune an exploratory analysis and then call the result unsupervised discovery.',
'Centroid=The mean position representing a K-means cluster.|Density cluster=A group defined by nearby populated regions.|Noise point=An observation not assigned to a density cluster.|Silhouette=A measure comparing within-cluster closeness and separation.',
'''from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X = load_wine().data # Use measurements without their labels.
Z = StandardScaler().fit_transform(X) # Fit an exploratory representation.
labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(Z) # Group rows into three clusters.
print("Cluster sizes:", [(labels == i).sum().item() for i in range(3)]) # Inspect representation.
print("Silhouette:", round(silhouette_score(Z, labels), 3)) # Evaluate geometric separation.
assert len(set(labels)) == 3 # Verify the requested grouping was formed.
''',
'Cluster numbers are arbitrary IDs, not automatically wine classes. This is exploratory clustering of the supplied table, not held-out predictive evaluation. A good geometric score does not prove the groups are meaningful for a decision.',
'Compare scaled and unscaled K-means, then DBSCAN. Report cluster sizes and noise counts and explain how interpretation changes.',
'Use the same input features and record preprocessing. Exclude or account for DBSCAN noise labels when interpreting clustering metrics; do not equate label -1 with a discovered target class.',
'Are cluster numbers automatically the known class labels?', 'Yes; cluster 0 must equal target class 0.', 'No. Cluster IDs are arbitrary and grouping depends on representation, algorithm, and settings.')
L('Dimensionality reduction and feature selection',
'Feature selection retains a subset of original inputs. Dimensionality reduction constructs new combinations. PCA finds orthogonal directions of large variance after centering; high variance does not guarantee predictive usefulness. TruncatedSVD can work efficiently with sparse text matrices without centering them.',
'A summary can retain original sentences or rewrite their ideas into a shorter form.',
'Fit selection or reduction inside cross-validation when evaluating a predictor. Scale before PCA when units should not determine variance. Explained variance describes retained spread, not accuracy. Compare a reduced pipeline against a suitable unreduced baseline and inspect whether smaller size or faster prediction justifies the tradeoff.',
'Component=A new feature formed from original features.|Explained variance=The amount of data spread captured by components.|Feature selection=Choosing original input columns.|PCA=A linear projection onto directions of large centered-data variance.',
'''from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
X = load_digits().data # Each image is represented by 64 pixel features.
prep = make_pipeline(StandardScaler(), PCA(n_components=10)) # Scale and reduce dimensionality.
Z = prep.fit_transform(X) # Build an exploratory ten-dimensional representation.
print("Original / reduced shapes:", X.shape, Z.shape) # Inspect compression.
print("Variance retained:", round(prep.named_steps["pca"].explained_variance_ratio_.sum(), 3)) # Sum retained variance ratios.
assert Z.shape[1] == 10 # Check component count.
''',
'Ten components replace 64 original features in this representation. The retained variance fraction is not a fraction of classification accuracy. This example explores the table; predictive comparison would fit the whole transformation within training folds.',
'Compare a classifier pipeline with and without PCA on the same folds. Record score and transformed feature count.',
'Use make_pipeline(StandardScaler(), PCA(...), classifier). Select the component count using development evaluation, not the final test.',
'Does retaining 95% of variance guarantee retaining 95% of accuracy?', 'Yes; variance and accuracy are interchangeable.', 'No. Predictive information may lie in low-variance directions; evaluate the complete predictive workflow.')
L('Multiclass, multilabel, and multioutput tasks',
'Multiclass classification chooses one of several labels. Multilabel classification can assign several labels to the same example. Multioutput regression predicts several numbers. Their target shapes and evaluation meanings differ. A wrapper can fit a compatible base estimator separately for each target, but this may ignore relationships among outputs.',
'A book has one shelf category, several topic tags, and several numerical measurements; these are different prediction tasks.',
'Inspect target meaning and shape before selecting a wrapper. A multilabel indicator table uses one column per label. Subset accuracy requires the entire predicted label set to match, which is stricter than per-label accuracy. Compare micro and macro metrics and support. For multioutput regression, decide how individual output errors should be aggregated and retain units.',
'Multiclass=Exactly one label from more than two possible classes.|Multilabel=Several labels may apply to one example.|Multioutput=More than one target value per example.|Subset accuracy=The fraction of examples with every label predicted correctly.',
'''from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
X, y = make_multilabel_classification(n_samples=120, n_features=10, n_classes=3, random_state=42) # Generate label-indicator targets.
Xt, Xv, yt, yv = train_test_split(X, y, random_state=42) # Create a development validation split.
model = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(Xt, yt) # Fit one binary predictor per label.
pred = model.predict(Xv) # Return one indicator per sample and label.
print("Target shape:", y.shape) # Three possible labels per example.
print("Micro F1:", round(f1_score(yv, pred, average="micro"), 3)) # Aggregate label decisions.
print("Subset accuracy:", round(accuracy_score(yv, pred), 3)) # Require every label to match.
assert pred.shape == yv.shape # Preserve output dimensions.
''',
'The two scores judge different properties and need not be equal. A two-dimensional target is not automatically multilabel: determine whether columns are indicator labels or independent numerical outcomes.',
'Construct one example target for each of multiclass, multilabel, and multioutput regression. Explain why blindly flattening a two-dimensional target loses meaning.',
'Multiclass might be [0,2,1]; multilabel might be [[1,0,1],[0,1,0]]; multioutput regression might be [[12.5,3.1],[9.2,4.8]]. Preserve one row per sample.',
'When does subset accuracy count a multilabel prediction as correct?', 'When at least one predicted label is right.', 'It requires all labels for that sample to match, so it is stricter than individual label correctness.')
L('Introductory model inspection',
'Inspection asks how a fitted model uses inputs. Permutation importance measures how an evaluation score changes when a feature’s values are shuffled. Partial dependence averages model predictions as a selected feature is varied. Neither method proves causality, and correlated features can make interpretation difficult.',
'Covering one part of a map shows how much a navigator relied on it, not what caused the roads to exist.',
'Use an appropriate held-out development partition for inspection when results guide improvements. Repeat permutations to inspect variation. Correlated substitutes can make each feature appear individually less important. Partial dependence can evaluate unrealistic feature combinations when inputs are strongly related. Report the model, scoring rule, data, and these limitations.',
'Permutation importance=Score change after shuffling a feature during evaluation.|Partial dependence=An average prediction response to changing selected feature values.|Correlated features=Inputs whose values vary together.|Causal claim=A statement about what changing a cause would do, requiring more than model association.',
REG+'''from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, partial_dependence
Xt, Xv, yt, yv = train_test_split(X_dev, y_dev, random_state=7) # Development-only inspection split.
model = RandomForestRegressor(n_estimators=30, random_state=42).fit(Xt, yt) # Fit a small inspectable model.
result = permutation_importance(model, Xv, yv, n_repeats=3, random_state=42, scoring="neg_mean_absolute_error") # Repeat feature shuffles.
print("Mean importances:", result.importances_mean.round(2)) # Positive means shuffling worsened score on average.
pd_result = partial_dependence(model, Xv, features=[0], grid_resolution=5) # Average predictions over a feature grid.
print("Partial dependence shape:", pd_result["average"].shape) # Inspect response dimensions.
assert len(result.importances_mean) == X.shape[1] # One importance per input feature.
''',
'Importances are score differences under the chosen evaluation. Negative values can occur from variability or harmful features. For a plot use PartialDependenceDisplay.from_estimator; inspect feature dependence before interpreting the curve.',
'Write two sentences describing a feature’s model importance and one sentence explaining why it is not a causal conclusion.',
'Name the score, dataset partition, and repeat variation. Explain that shuffled or intervened feature combinations may not correspond to realistic observations.',
'Does a large permutation importance prove a causal effect?', 'Yes; shuffling is a real-world causal experiment.', 'No. It measures dependence of a fitted model’s score on a feature under a particular perturbation.')
L('Communicating findings',
'A useful report connects the prediction problem, evidence, and decision. Explain the baseline, appropriate evaluation split, metric units, uncertainty, and remaining weaknesses. Avoid claiming that a tiny score improvement is meaningful without checking its variability and operational effect.',
'A model report is a handover note: another person should understand what was done and what to trust.',
'Write six short sections: problem and prediction time; data and representativeness; evaluation design; baseline and candidate results; errors and subgroup checks; recommendation and limitations. Distinguish observations from hypotheses. For small subgroups report counts and uncertainty. Name what the model cannot establish and what additional evidence is needed.',
'Finding=A claim directly supported by the analysis.|Limitation=A boundary on what the evidence supports.|Recommendation=A proposed action justified by evidence and tradeoffs.|Reproducibility=The ability to rerun a documented process and obtain consistent results.',
'',
'A clear sentence names the comparison and its context: “On four grouped development folds, the selected pipeline reduced mean absolute error from A to B minutes; variation remained high for group C.” Replace letters only with executed results. Do not invent numbers to make the report look complete.',
'Write a one-page report for an earlier project. Include one finding, one uncertainty, one subgroup check with counts, and one bounded recommendation.',
'Self-review with four questions: Are claims tied to results? Are units and partitions named? Are limitations specific? Can another person locate the code and data instructions? Revise any unsupported claim.',
'Which statement belongs in a defensible report?', 'This model is perfect because its training score is high.', 'Describe held-out evidence, baseline comparison, uncertainty, intended use, and remaining limitations.', 'Unsupported conclusion','Trace each conclusion to a table, plot, or documented fact; revise claims that exceed the evidence.')
L('Saving and loading complete pipelines',
'A model file must preserve the preprocessing needed for predictions. Saving only the classifier loses learned scaling, encodings, and column handling. Pickle-based formats such as joblib can execute code when loaded, so load only artifacts from sources you trust. Compatibility across arbitrary library versions is not guaranteed.',
'Saving a cake recipe without the ingredient preparation instructions does not preserve the full process.',
'Fit the selected complete pipeline, record dependency versions, and serialize it. Reload it and compare predictions on a fixed input. Keep the input schema and model metadata alongside it. This chapter checks artifact round-tripping; Chapter 35 checks input validation and a clean reproducible workflow.',
'Persistence=Saving a fitted object for later reuse.|Artifact=A saved output such as a trained pipeline file.|Round-trip check=Comparing behavior before saving and after reloading.|Trusted source=An origin whose executable artifacts you are willing to load.',
CLASS+'''import joblib
import numpy as np
model.fit(X_dev, y_dev) # Fit a complete pipeline.
before = model.predict_proba(X_test[:5]) # Capture a fixed prediction reference.
joblib.dump(model, "course_pipeline.joblib") # Save only to a file you control.
loaded = joblib.load("course_pipeline.joblib") # Load this known, self-created artifact.
after = loaded.predict_proba(X_test[:5]) # Repeat the same input.
print("Predictions match:", np.allclose(before, after)) # Check numeric consistency.
assert np.allclose(before, after) # Fail if the round trip changes predictions.
''',
'The true condition verifies the save/load round trip in this environment, not compatibility with every future package release. The reference predictions are not a new model-selection score.',
'Save a mixed-data pipeline from an earlier project and record Python and package versions. Reload it and compare predictions on a fixed raw record.',
'Use the complete Pipeline object. Keep requirements.txt and input schema with the model, and compare probabilities or numeric predictions with a justified tolerance.',
'Why not load an arbitrary joblib file from an unknown source?', 'It contains only harmless plain-text coefficients.', 'Loading pickle-based artifacts can execute code. Trust and provenance matter before loading.')
L('Reproducible prediction workflows',
'A reusable workflow checks what it receives. Missing columns, extra columns, wrong units, and invalid types can silently undermine predictions. Reproducibility also needs data access instructions, dependency records, seeds, and clear execution order. Predicting from a fresh process is stronger evidence than rerunning one cell in a long-lived notebook.',
'A plug fits reliably only when the socket has the expected shape and voltage.',
'Define required columns and permitted types or values. Decide whether extra columns are rejected or ignored. Distinguish a deliberately missing value from an unparseable value. Write a small prediction script that validates before calling the loaded pipeline. Check it from a fresh environment using documented commands and an example input.',
'Input schema=A contract for column names, types, units, and allowed values.|Validation=A check that input meets a specified contract.|Seed=A value controlling repeatable random choices.|Clean run=Execution without relying on hidden prior state.',
'''import pandas as pd
required = ["hours", "distance"] # Define expected input columns.
def validate(frame):
    if set(frame.columns) != set(required): # Reject missing or unexpected columns.
        raise ValueError("Expected exactly hours and distance") # Explain the contract.
    result = frame[required].apply(pd.to_numeric, errors="raise") # Convert valid numeric values in fixed order.
    if result.isna().any().any(): # This simple schema disallows missing values.
        raise ValueError("Missing numeric input") # Reject incomplete records clearly.
    return result # Return a validated ordered table.
print(validate(pd.DataFrame({"distance": [3], "hours": [2]})).to_dict("records")) # Demonstrate order normalization.
try:
    validate(pd.DataFrame({"hours": [2]})) # Deliberately omit a required feature.
except ValueError as error:
    print("Rejected:", str(error)) # Show a useful validation message.
assert list(validate(pd.DataFrame({"hours": [2], "distance": [3]})).columns) == required # Check order.
''',
'The valid record is returned in a fixed column order. The invalid record is rejected before prediction. This illustrative schema is stricter than a pipeline designed to impute missing values; a real contract should match the trained workflow.',
'Write predict.py for your saved pipeline, plus a valid example input and two invalid cases. Document one command that reproduces predictions.',
'Validate columns and types, load only your trusted artifact, call predict, and return understandable output. Keep validation rules consistent with training preprocessing.',
'Does setting random_state alone make an entire project reproducible?', 'Yes; no environment or data records are needed.', 'No. Seeds help, but data versions, dependency versions, input contracts, and execution instructions also matter.')
L('Guided capstone',
'The capstone combines decisions that earlier lessons isolated. Choose a prediction question with a clear target, a practical baseline, and a split matching the deployment scenario. Good work may conclude that the available data cannot support the intended decision. The deliverable is a reproducible investigation, not the highest possible leaderboard score.',
'Building a complete bicycle tests whether the parts work together, not just whether each part looks polished.',
'Checkpoint 1: document prediction time, data provenance, and candidate leakage. Checkpoint 2: reserve the final test and establish a baseline on development folds. Checkpoint 3: construct preprocessing and compare two candidates. Checkpoint 4: tune a bounded search and inspect errors. Checkpoint 5: freeze decisions, evaluate once on the reserved test, save the pipeline, and write the report and reproduction instructions. The chapter’s 6–10 hour budget includes all deliverables.',
'Capstone=An integrated project applying previously learned skills.|Prediction time=The moment at which available inputs must produce an answer.|Frozen decision=A modeling choice finalized before test evaluation.|Handover=A usable package of code, artifacts, instructions, and findings.',
REG+'''from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
candidates = {"baseline": DummyRegressor(), "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1))} # Prespecified candidates.
errors = {} # Collect development evidence.
for name, estimator in candidates.items():
    errors[name] = -cross_val_score(estimator, X_dev, y_dev, cv=4, scoring="neg_mean_absolute_error").mean() # Evaluate without final-test selection.
winner = min(errors, key=errors.get) # Select lower development MAE.
selected = candidates[winner].fit(X_dev, y_dev) # Fit the frozen choice on development data.
print("Development MAE:", {k: round(v, 2) for k, v in errors.items()}) # Explain selection evidence.
print("Selected:", winner) # Record the frozen candidate.
print("Final-test MAE:", round(mean_absolute_error(y_test, selected.predict(X_test)), 2)) # One final evaluation.
assert len(selected.predict(X_test)) == len(y_test) # Check output alignment.
''',
'This small synthetic skeleton illustrates decision order. Your capstone uses a real dataset and includes a full audit, report, and saved pipeline. Do not loop back to tuning because the final-test score was disappointing; report the finding and plan a new evaluation if development continues.',
'Complete the capstone project brief. Save a pipeline and write a report that separates cross-validation selection from final-test evaluation.',
'Use the module project rubric and checkpoints. A complete solution includes dataset instructions, code, a saved artifact, pinned dependencies, error analysis, limitations, and reproduction commands.',
'What should happen before the final test is evaluated?', 'Repeatedly adjust settings using test outcomes.', 'Freeze the modeling and decision rules using development evidence, then evaluate the selected workflow.')
L('Diagnose a flawed project',
'A convincing number can come from a broken workflow. Diagnose the information flow before changing algorithms. Common mistakes include fitting a scaler on all rows, putting a target-revealing column in X, using the wrong split, optimizing the wrong metric, and choosing settings from final-test performance.',
'Inspect the wiring before replacing the light bulb.',
'Repair each issue separately and explain why it invalidated the original claim. The example below shows the corrected sequence; the exercise asks you to describe the broken alternative. A higher repaired error can be a sign of more honest evaluation rather than a worse engineering result.',
'Audit trail=A record of decisions and repairs with their reasons.|Leak source=The path by which unavailable information entered development.|Metric mismatch=An evaluation rule that does not reflect the intended task.|Unsupported conclusion=A claim not justified by the observed evidence.',
CLASS+'''from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(3, shuffle=True, random_state=42) # Choose folds before comparing settings.
scores = cross_val_score(model, X_dev, y_dev, cv=cv, scoring="accuracy") # Fit the scaler inside each training fold.
print("Repaired development accuracy:", round(scores.mean(), 3)) # Label the result honestly.
print("Final-test rows remain reserved:", len(X_test)) # Count but do not use test outcomes.
assert len(scores) == 3 # Check evaluation structure.
''',
'The corrected code excludes test data and evaluates the full pipeline. Its score alone cannot prove that features are available at prediction time; inspect column meanings and data collection as well.',
'Write five planted mistakes in a copy of an earlier notebook, then repair them. Include one feature leak, one split error, one metric error, one preprocessing error, and one unsupported conclusion.',
'For each repair record: faulty line, invalid assumption, corrected code, and affected claim. Never submit a knowingly leaking score as the repaired model’s performance.',
'If fixing leakage lowers the score, was the repair necessarily wrong?', 'Yes; valid repairs always improve the score.', 'No. Removing leaked information often lowers an inflated score while making the evaluation more credible.')
L('Independent assessment',
'An independent assessment removes the step-by-step recipe. You receive a new dataset, an outcome to predict, and a rubric. Your task is to explain and justify the process. A simpler well-evaluated model can be a stronger submission than a complex model with an unclear split or missing reproducibility instructions.',
'The final driving assessment checks whether you can choose the right action without someone narrating every turn.',
'Define the target and prediction moment before examining candidate features. Document the split, baseline, preprocessing, candidate comparison, tuning budget, error analysis, and final evaluation. Submit the same reproducibility package used in the capstone, but make the decisions yourself. The assessment budget is 6–10 hours and is provisional.',
'Independent assessment=A task completed without a prescribed sequence of modeling steps.|Rubric=A set of explicit criteria for judging work.|Evidence=A result or documented fact supporting a decision.|Scope=A stated boundary on the problem and deliverables.',
'',
'There is no single expected score. A valid submission can explain why its model underperforms a baseline or why the data is insufficient. It must not hide negative findings or invent results.',
'Complete the independent module brief with a dataset not used for your capstone. Include a decision log explaining at least five choices.',
'Use the four rubric dimensions. Check every output can be recreated, every metric names its partition, and every claim stays within the evidence. Compare your work to the rubric rather than copying a worked solution.',
'Can a project that fails to beat its baseline still demonstrate useful learning?', 'No; only a high score can pass any criterion.', 'Yes, if the code, evaluation, explanation, and reproducibility are sound and the conclusion honestly reflects the evidence.', 'No automatic correctness check for a written report','Use the explicit rubric and, when possible, a second reader. Self-assessment is not independent grading.')
L('Additional SVM and Naive Bayes models',
'A support vector machine seeks a boundary with a useful margin between classes. A kernel lets it represent nonlinear relationships without explicitly constructing every feature combination. C controls the penalty for margin violations. Different Naive Bayes variants make different assumptions about feature distributions; choose them according to representation, not just naming.',
'A wide lane between groups is easier to follow than a boundary squeezing between every point.',
'Scale numerical inputs before SVC or SVR unless a justified alternative representation is used. RBF kernels introduce gamma, which controls locality. Large datasets can make kernel methods expensive. GaussianNB models continuous class-conditional features, while MultinomialNB is suited to nonnegative count-like representations. Conditional independence is a simplifying assumption, not a statement that real features are truly unrelated.',
'Margin=A separation region associated with a fitted SVM boundary.|Kernel=A function defining similarity in an implicit feature space.|C=An SVM setting controlling the cost of margin violations.|Conditional independence=An assumption that features are independent given a class in Naive Bayes.',
CLASS+'''from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
candidate = make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", gamma="scale")) # Scale before a nonlinear margin model.
scores = cross_val_score(candidate, X_dev, y_dev, cv=3, scoring="accuracy") # Compare through development validation.
print("Mean accuracy:", round(scores.mean(), 3)) # Summarize performance.
assert len(scores) == 3 # Verify evaluation count.
''',
'The kernel and its settings define flexibility. This score does not establish that an RBF kernel is better than a linear model; compare both on identical splits and record compute cost.',
'Compare a linear SVC pipeline, an RBF SVC pipeline, and GaussianNB on numerical data. Explain which assumptions differ.',
'Use shared folds and metrics. Record scaling for SVMs and the Gaussian class-conditional assumption for GaussianNB.',
'Which setting governs locality for an RBF SVM?', 'The target array shape alone.', 'Gamma affects the locality of the RBF similarity; evaluate it together with C.')
L('Anomaly detection',
'An unusual record is not automatically an error or fraud. Anomaly detection identifies deviations relative to a learned reference. Outlier detection often describes unusual training records; novelty detection evaluates new records against a reference fitted on normal examples. The distinction affects which estimator methods are appropriate.',
'A librarian notices a book shelved far from similar books, then investigates rather than declaring it counterfeit.',
'IsolationForest isolates unusual records through randomized partitions. LocalOutlierFactor compares local densities; its default outlier workflow differs from novelty=True use for new examples. contamination affects an operating threshold, not a guaranteed true anomaly proportion. Validate flagged records with domain evidence and estimate the review burden.',
'Anomaly=An observation unusual relative to a chosen reference.|Novelty detection=Finding unusual new examples against a fitted reference.|Contamination=A setting describing an assumed outlier fraction or threshold policy.|Review burden=The number of flagged cases requiring investigation.',
'''import numpy as np
from sklearn.ensemble import IsolationForest
rng = np.random.default_rng(42) # Reproducible numerical examples.
X = np.r_[rng.normal(size=(60, 2)), [[10, 10]]] # Add a distant record to a reference cloud.
model = IsolationForest(contamination=0.05, random_state=42).fit(X) # Fit a demonstration detector.
labels = model.predict(X) # Negative one marks flagged records.
print("Flagged count:", int((labels == -1).sum())) # Report operational volume.
print("Distant record label:", int(labels[-1])) # Inspect the injected example.
assert set(labels) <= {-1, 1} # Check estimator label convention.
''',
'Flags indicate model-defined unusualness, not verified wrongdoing. The threshold can also flag ordinary tail examples. Report investigation outcomes separately from model scores.',
'Inject several kinds of anomalies and compare IsolationForest with LocalOutlierFactor. Explain whether you are detecting training outliers or new novelties.',
'Choose the appropriate LOF mode and evaluation records. Do not treat an arbitrary contamination fraction as a known ground-truth prevalence.',
'Does an anomaly flag prove fraud?', 'Yes; unusualness and fraud are the same label.', 'No. A flag is a prompt for investigation; verify with meaningful labels or domain review.')
L('Custom transformers',
'A custom transformer makes feature engineering reusable inside training folds. Constructor parameters should describe configuration; fit learns from training data; transform applies the learned rule. A transformer must preserve row alignment and predictable feature dimensions. Arbitrary functions that depend on held-out targets break the workflow.',
'A custom assembly-line station should accept the same input contract and produce consistent output every time.',
'Use FunctionTransformer for a simple stateless conversion. For a stateful rule, subclass BaseEstimator and TransformerMixin, expose constructor parameters, return self from fit, and validate input consistently. Test fitting and transforming separately. Names and persistence matter when loading a custom class from another process.',
'Stateless transformation=A mapping that does not learn data-dependent settings.|Stateful transformation=A mapping using values learned during fit.|TransformerMixin=A helper providing transformer behavior such as fit_transform.|Row alignment=Keeping output examples matched to the same input examples.',
'''import numpy as np
from sklearn.preprocessing import FunctionTransformer
X = np.array([[0, 3], [8, 15]]) # Nonnegative inputs.
transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True) # Define a stateless reversible mapping.
Z = transformer.fit_transform(X) # Fit the interface and apply the mapping.
print("Transformed:", Z.round(3)) # Inspect compressed values.
print("Recovered:", transformer.inverse_transform(Z).round(3)) # Reverse the mapping.
assert np.allclose(transformer.inverse_transform(Z), X) # Check this round trip.
''',
'FunctionTransformer delegates numerical work to the supplied function. The input domain must be valid for that function. This example is stateless; a learned clipping threshold would belong in fit of a stateful transformer.',
'Write a transformer that learns a training median and subtracts it. Confirm held-out rows never change the learned median.',
'Store median_ in fit, return self, and subtract median_ in transform. Use a constructor with explicit settings, and validate feature width before subtraction.',
'Where should a training-dependent statistic be learned?', 'In transform using every future input batch.', 'Learn it in fit on the current training partition and reuse it unchanged in transform.')
L('Larger datasets and incremental learning',
'A table that does not fit in memory requires a different workflow. Some estimators support partial_fit, updating a model batch by batch. This is not the same as fit repeatedly: ordinary fit usually starts a new learning process. Incremental learning also needs consistent features, suitable scaling, and an evaluation stream that avoids future-label leakage.',
'Reading a book one chapter at a time works only if the chapters share the same language and page order.',
'Use sparse storage for sparse features. HashingVectorizer creates a fixed feature space without learning a vocabulary but may have collisions. SGD estimators support incremental updates. For classification, provide all possible classes on the first partial_fit call. Track memory, batch order, and held-out performance. Cache only where repeated transformations are deterministic and beneficial.',
'Batch=A bounded subset processed at one time.|partial_fit=An incremental estimator update supported by some models.|Hashing collision=Different features mapping to the same hashed location.|Out-of-core learning=Training while loading only portions of data into memory.',
'''import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
X, y = make_classification(n_samples=200, n_features=8, random_state=42) # Small stand-in for a stream.
model = SGDClassifier(loss="log_loss", random_state=42) # An incremental classifier.
for start in range(0, 160, 40): # Train on four batches and reserve the last forty rows.
    model.partial_fit(X[start:start+40], y[start:start+40], classes=np.array([0, 1])) # Keep feature and class contracts fixed.
print("Held-out accuracy:", round(model.score(X[160:], y[160:]), 3)) # Evaluate unseen rows.
print("Seen training rows:", 160) # Record the training boundary.
assert model.n_features_in_ == 8 # Check consistent feature width.
''',
'The reserved rows did not update the model. A one-pass SGD result can be sensitive to learning rate, order, and scales; it is not a drop-in guarantee of a batch estimator’s performance.',
'Compare two batch sizes and record elapsed time, memory, and held-out results. Explain why labels from the future cannot update an earlier evaluation snapshot.',
'Keep the same evaluation boundary and feature mapping. Use time-aware test logic for real streams and make the availability of outcomes explicit.',
'Is calling fit on each batch generally equivalent to partial_fit?', 'Yes; both always accumulate earlier learning.', 'No. fit commonly resets an estimator, whereas partial_fit is the supported incremental-update interface.')
L('Ensembles and advanced evaluation',
'Voting combines model predictions. Stacking trains a meta-model using predictions from other models, which must be generated without allowing each training example to be predicted by a base model that fitted it. Nested cross-validation separates hyperparameter selection in inner folds from evaluation in outer folds.',
'A judge should evaluate a recipe-selection process using dishes the selection panel did not already optimize against.',
'Use built-in stacking support for out-of-fold meta-features rather than training a meta-model on overfitted base predictions. Compare an ensemble with its strongest individual model. Nested evaluation estimates the selection procedure and costs many fits. Similar component errors can limit ensemble gains. A final reserved test can still be useful after selecting a process.',
'Voting=Combining predicted labels or probabilities from multiple models.|Stacking=Learning a combination from base-model prediction features.|Out-of-fold prediction=A prediction for a row from a model fitted without that row.|Nested cross-validation=Inner model selection enclosed by separate outer evaluation folds.',
CLASS+'''from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
ensemble = VotingClassifier(estimators=[("linear", model), ("forest", RandomForestClassifier(n_estimators=25, random_state=42))], voting="soft") # Average class probabilities.
scores = cross_val_score(ensemble, X_dev, y_dev, cv=3, scoring="accuracy") # Refit components inside each fold.
print("Ensemble mean:", round(scores.mean(), 3)) # Compare against individual models separately.
assert len(scores) == 3 # Check fold count.
''',
'Soft voting requires component probability estimates and compatible class handling. A higher ensemble mean on this demonstration is not a universal advantage. Consider model size and latency as well.',
'Compare the ensemble and its components on identical folds. Sketch which data each stage of nested cross-validation may see.',
'The inner loop selects settings using outer-training rows only. The outer validation fold scores the resulting selection procedure without entering the search.',
'Why use out-of-fold predictions to teach a stacking meta-model?', 'To let the base models memorize each meta-training row first.', 'To avoid feeding the meta-model unrealistically optimistic in-sample base predictions.')
L('Time-series features and evaluation',
'Forecasting features must use information available at the forecast origin. A lag uses an earlier value. A rolling average must be shifted when the current target would otherwise enter the feature. The forecast horizon determines which past values are truly known at prediction time. Random row shuffling can hide these constraints.',
'Predict tomorrow’s weather from yesterday’s records, not from a report filed tomorrow evening.',
'Sort and validate timestamps first. Construct shifted lags and rolling windows for a stated horizon. Drop initial rows without sufficient history using aligned inputs and targets. Compare with a naive last-value or seasonal baseline. Use walk-forward validation and an appropriate gap when windows or labels overlap. Multi-step forecasting may need recursive predictions rather than future actual values.',
'Lag=A feature from an earlier time point.|Rolling window=A summary over a moving range of observations.|Forecast horizon=How far ahead the target occurs relative to the prediction origin.|Walk-forward evaluation=Repeated training on earlier data and validation on later periods.',
'''import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
series = pd.Series(np.arange(20, dtype=float)) # Ordered observations for an illustrative one-step forecast.
frame = pd.DataFrame({"target": series, "lag1": series.shift(1), "mean3": series.shift(1).rolling(3).mean()}).dropna() # Exclude the current target from features.
print(frame.head(3).to_string(index=False)) # Inspect timing manually.
for train, val in TimeSeriesSplit(3).split(frame): # Keep validation later than training.
    assert train[-1] < val[0] # Check temporal ordering.
print("One-step rows:", len(frame)) # Three early rows lack history.
assert frame.iloc[0]["lag1"] == 2 # The first complete target is 3, with lag 2.
''',
'This is a one-step feature example: each row assumes previous actual observations are available. It cannot be reused unchanged to claim a fixed-origin multi-step forecast where future lag values are not yet known.',
'Build a one-step bike-demand forecast with a last-value baseline. State forecast origin, horizon, timestamp frequency, and gap policy.',
'Shift before rolling and evaluate in time order. For longer horizons, define which historical values and external forecasts are available at each prediction origin.',
'Why shift before computing a target rolling mean for a one-step predictor?', 'To include the answer in the input feature.', 'To ensure the current target does not enter its own predictor; features must use available past observations.')
L('External boosting libraries',
'External boosting libraries offer estimators with familiar fit and predict methods, but their settings and early-stopping interfaces differ. Compatibility is useful, not a promise that one library’s parameters can be copied into another. This lab uses XGBoost’s CPU package and compares its role with LightGBM and CatBoost.',
'Different vehicles share steering wheels but have different controls and maintenance requirements.',
'XGBoost supports tree boosting and regularization; LightGBM emphasizes efficient histogram-based training; CatBoost includes specialized categorical-feature handling. Native category handling is library-specific. Compare one detailed implementation with the core HistGradientBoosting baseline on identical splits. Read the installed library’s early-stopping API and keep evaluation sets inside training-fold boundaries.',
'Wrapper API=An interface adapting a library to familiar estimator conventions.|Boosting round=One stage adding to a boosted prediction function.|Native categorical handling=Library-specific learning directly from categorical representations.|External dependency=A separately installed package outside scikit-learn.',
'''from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
X, y = make_classification(n_samples=180, n_features=8, random_state=42) # Bounded demonstration data.
Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, random_state=42) # Development validation only.
model = XGBClassifier(n_estimators=20, max_depth=3, learning_rate=0.1, tree_method="hist", n_jobs=1, random_state=42) # Configure a CPU histogram model.
model.fit(Xt, yt) # Fit using training rows.
print("Validation accuracy:", round(model.score(Xv, yv), 3)) # Evaluate on held-out development examples.
assert len(model.predict(Xv)) == len(yv) # Verify output count.
''',
'This sample uses xgboost-cpu, which exposes the xgboost import. It does not need a GPU. The score is a development result; compare with an equal-budget baseline before recommending a library.',
'Run the same bounded problem with HistGradientBoostingClassifier. Read LightGBM and CatBoost documentation and compare categorical handling, installation, and early stopping.',
'Keep the split, metric, and practical time budget fixed. Do not pass a global final-test set as eval_set during tuning. Follow each library’s installed-version documentation.',
'Can early-stopping settings be copied blindly between boosting libraries?', 'Yes; compatible fit means identical parameter semantics.', 'No. Interfaces and training semantics differ; verify the installed version and preserve validation boundaries.')
L('Resampling with imbalanced-learn',
'Resampling changes the training examples or their proportions. Random oversampling repeats rows, undersampling removes rows, and SMOTE interpolates synthetic examples between minority neighbors. Synthetic examples are not newly observed evidence. Resampling can amplify noise or generate implausible combinations, especially with unsuitable feature representations.',
'Repeating a rare chapter in your practice schedule changes emphasis without creating new independent books.',
'Use an imbalanced-learn Pipeline so sampling happens only within each cross-validation training fold. Keep validation and test class distributions representative. Fit scaling and encodings inside the same evaluation boundary. Compare weighting and thresholds before assuming SMOTE helps. SMOTE needs enough minority examples for its neighbor setting; categorical data may require a different sampler.',
'Oversampling=Increasing training representation of selected classes.|Undersampling=Reducing training representation of selected classes.|SMOTE=Creating synthetic minority examples using neighbor interpolation.|Sampling pipeline=A workflow that restricts resampling to fitting partitions.',
'''from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
X, y = make_classification(n_samples=400, n_features=8, weights=[0.9, 0.1], random_state=42) # Preserve a natural validation imbalance.
model = make_pipeline(StandardScaler(), SMOTE(random_state=42), LogisticRegression(max_iter=1000)) # Sample inside fitting only.
cv = StratifiedKFold(3, shuffle=True, random_state=42) # Shared class-aware partitions.
scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision") # Evaluate on unsampled validation rows.
print("Average precision:", round(scores.mean(), 3)) # Compare with nonsampling baselines separately.
assert len(scores) == 3 # Confirm evaluation rounds.
''',
'The sampler is inside the fitted workflow; validation rows are not balanced by the sampler. An imbalanced-learn pipeline has sampling-specific semantics, including a fit_transform distinction, so avoid assuming every transformer contract is identical.',
'Compare no resampling, class weighting, and SMOTE on identical folds. Report prevalence, average precision, and a development threshold analysis.',
'Use the same evaluation distribution and keep each intervention within fitting partitions. Do not oversample the full dataset before splitting.',
'Where should SMOTE run during cross-validation?', 'Once on the complete dataset before forming folds.', 'Inside each training fold through a sampling-aware pipeline, leaving validation and final-test examples untouched.')
L('Minimal API serving and monitoring',
'A prediction API receives structured input, validates it, calls a trusted fitted workflow, and returns structured output. A static course page does not host a Python service. This lab is a local demonstration; production deployment adds authentication, capacity, dependency, logging, and operational requirements.',
'A service counter checks the order before sending it to the kitchen and returns a clearly labeled result.',
'Create a minimal FastAPI endpoint with a Pydantic request schema. Load the pipeline from a trusted artifact at startup for a real service. Reject invalid requests clearly and keep units explicit. Monitor input distributions, missingness, latency, and failure rates. Input drift is a change in inputs; it does not by itself prove accuracy fell. Performance monitoring needs representative outcomes when they arrive.',
'Endpoint=A URL and HTTP method handling a particular request.|Schema validation=Checking request fields, types, and constraints.|Input drift=A change in incoming feature distributions.|Performance monitoring=Tracking predictive quality using appropriate observed outcomes.',
'''from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit([[1], [2], [3]], [3, 5, 7]) # A tiny demonstration estimator.
app = FastAPI() # Create a local API application.
class Request(BaseModel):
    hours: float = Field(ge=0, le=24) # Require a bounded numerical input.
@app.post("/predict")
def predict(request: Request):
    return {"prediction": float(model.predict([[request.hours]])[0])} # Return JSON-compatible output.
client = TestClient(app) # Exercise the local endpoint without a live server.
response = client.post("/predict", json={"hours": 4}) # Send valid input.
print("Valid:", response.status_code, response.json()) # Inspect HTTP status and result.
print("Invalid status:", client.post("/predict", json={"hours": -1}).status_code) # Demonstrate schema rejection.
assert response.status_code == 200 # Verify successful local serving behavior.
''',
'HTTP 200 indicates success; 422 indicates request validation failure in this example. To run a saved version as a service, place it in api.py and start uvicorn api:app --reload for local development. The course website itself serves educational files, not this Python endpoint.',
'Build a local prediction endpoint for an earlier saved pipeline. Define three monitoring indicators and explain which require true outcomes.',
'Input distribution and failure-rate checks need requests; accuracy or calibration needs labels or numerical outcomes. Keep sensitive raw inputs out of logs unless collection is justified and controlled.',
'Does detecting input drift prove predictive performance has degraded?', 'Yes; any distribution change equals lower accuracy.', 'No. Drift is a warning signal; evaluate outcomes when available to establish changes in predictive quality.')
assert len(LESSONS)==47, len(LESSONS)
PROJECTS=[
('Repeat the five moves','Wine','https://scikit-learn.org/1.8/datasets/toy_dataset.html#wine-recognition-dataset','Use load_wine(return_X_y=True). This built-in dataset has 178 examples, 13 numeric features, and three classes. Read the dataset description and attribution before redistributing it.','A top-to-bottom notebook; input and target shapes; a majority baseline; a scaled logistic classifier; two accuracies; five sentences explaining the workflow; one controlled split change.','Load and inspect → make a stratified split → fit the baseline on training labels → fit the pipeline → report the demonstration holdout results. Change test size as a learning experiment, explicitly treating these runs as development rather than an untouched final estimate.'),
('Become a data detective','Titanic / OpenML 40945','https://www.openml.org/d/40945','Use fetch_openml(data_id=40945, as_frame=True). Inspect its OpenML record, original source, and usage conditions before download or redistribution. Reserve a final partition before target-informed exploration.','A data dictionary; five labeled plots with observations; at least five modeling issues; an explicit list of outcome-revealing columns; source and usage notes.','Inspect column meanings, dtypes and missingness. Investigate boat/body-related fields for outcome leakage and identify which measurements existed at the prediction moment. This is an audit task: do not manufacture predictive metrics to satisfy a rubric.'),
('Repair the leaky notebook','Breast cancer diagnostic','https://scikit-learn.org/1.8/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset','Use load_breast_cancer. Data is available in scikit-learn; inspect DESCR for source and attribution. The task is educational classification, not a clinical validation.','A corrected notebook; a table of leaks and fixes; a baseline and pipeline compared on the same stratified development folds; a reserved final test.','Start from the flawed-notebook download. Move scaling into a pipeline, move setting selection to development folds, document every repair, and leave the final-test answers unused during selection.'),
('Tell two error stories','California housing + handwritten digits','https://scikit-learn.org/1.8/datasets/real_world.html#california-housing-dataset','Housing: fetch_california_housing; inspect its documentation and original source. Digits: load_digits. Downloads may need internet. Record target units and dataset provenance.','Two reports, each with baseline and model; appropriate splits; regression MAE/RMSE and a residual plot; classification metrics and confusion matrix; the most confused distinct digit pair.','For housing, explain errors in original dollar units after checking the target’s scaling. For digits, inspect off-diagonal matrix counts. Select candidate choices on development data before final reporting.'),
('A pipeline that accepts raw rows','Ames housing / OpenML 42165','https://www.openml.org/d/42165','Use fetch_openml(data_id=42165, as_frame=True). Inspect the record and source licence/usage conditions. Review identifiers, missing-value semantics and the meaning of each category.','One raw-record pipeline; numeric and categorical preprocessing; a target-transformation comparison; original-unit evaluation; successful predictions for a missing value and an unseen category.','Separate X and y, reserve the final test, build ColumnTransformer branches, wrap regression with TransformedTargetRegressor when justified, and compare original-unit development errors. Demonstrate input robustness with constructed rows.'),
('Choose a model fairly','Adult census income / OpenML 1590','https://www.openml.org/d/1590','Use fetch_openml(data_id=1590, as_frame=True). Inspect original UCI documentation and usage terms. Select a documented, reproducible subset if compute is limited.','Four candidates: neighbors, a tree, a forest, histogram boosting; common folds and metric; score/spread, fit-time and serialized-size table; a justified recommendation.','Keep all preprocessing within each candidate pipeline. Use comparable evaluation and report resource measurements as environment-dependent. Avoid treating historical income labels as a neutral or complete measure of people.'),
('Make rare events count','Credit-card fraud / OpenML 1597','https://www.openml.org/d/1597','Use fetch_openml(data_id=1597, as_frame=True) after checking its record and usage conditions. Verify target coding and stratify any bounded subset. State the resulting prevalence.','A metric justification; class-weight comparison; development-only threshold selection; a tuned pipeline; final reserved-test result; review burden and remaining weaknesses.','Preserve rare positives in your subset. Compare with a dummy rule, select using average precision and an operational threshold objective, then freeze the rule before final testing. Do not resample evaluation rows.'),
('Three ways to learn','20 Newsgroups, Wine, generated multilabel','https://scikit-learn.org/1.8/datasets/real_world.html#the-20-newsgroups-text-dataset','Fetch four newsgroup categories with remove=("headers","footers","quotes") and record the exact names. Use load_wine for clustering and make_multilabel_classification for a bounded label exercise. Read source descriptions before reuse.','Three text models compared inside text pipelines; scaled/unscaled Wine cluster reports; a multilabel exercise with target shape and metrics; notes distinguishing their evaluations.','Reserve text holdouts. Treat clustering as exploration and explain geometric scores without equating clusters to truth. Use a separate held-out split for generated multilabel evaluation.'),
('Package work someone can use','Your Module 5 or 6 pipeline','https://scikit-learn.org/1.8/model_persistence.html','Reuse your own trusted fitted artifact and data instructions. Do not distribute restricted datasets; provide access instructions instead.','Saved complete pipeline; prediction script; schema; requirements; example input; README; findings report; before/after-load prediction check.','Ask a second person to run the instructions in a clean environment where possible. Record their result honestly. Self-checking on your own machine is useful but is not an independent reproduction.'),
('Your independent solution','A new documented dataset','https://www.openml.org/search?type=data','Choose a dataset not used in your guided capstone. Confirm source, usage terms, target meaning, and sensible splits before modeling. Keep the problem bounded.','Chapter 36: guided report and artifact; Chapter 37: repaired notebook and explanations; Chapter 38: independently chosen workflow, comparison, error analysis and limitations; final test.','Complete the guided capstone checkpoints, repair the supplied planted mistakes, then solve a fresh task using the rubric without step-by-step directions. Do not reuse final-test outcomes for tuning.'),
('Choose your next adventure','Optional labs','https://scikit-learn.org/1.8/user_guide.html','Each extension has its own lab and dependency needs. See the lesson exercise and the optional requirements file.','One focused lab and a five-question quiz for each chosen extension.','Optional chapters never gate the core path. Record assumptions, reproducibility instructions and any limits of your chosen extension.')]
CHALLENGE=[('Regression baseline','7–11','Wine is classification; instead use load_diabetes for this regression task. Compare a mean baseline and a linear model with development MAE. Reserve a final test.'),('Multiple-feature regression','10–11, 19','Use load_diabetes. Compare a small justified feature subset with all features using Ridge. Keep folds fixed and explain the result.'),('Mixed-data pipeline','14–18','Use a bounded Ames sample. Build numeric and categorical branches and predict one raw record containing a missing value.'),('Classification errors','12–13','Use load_digits. Train a scaled logistic classifier and explain the largest off-diagonal confusion counts.'),('Rare events','25–27','Use make_classification with 5% positives. Compare weighting and thresholds using development precision and recall.'),('Words into predictions','28','Use four 20 Newsgroups categories. Compare TF-IDF plus LinearSVC with a Naive Bayes baseline.'),('Fair model comparison','9, 20–22','Use load_wine. Compare a tree, forest, and neighbors on identical stratified folds and report variability.'),('A small tuning budget','24–25','Use load_breast_cancer. Tune three regularization strengths inside a scaled pipeline and record the fit count.'),('Find groups','29','Use Wine inputs without labels. Compare scaled and unscaled clusters and explain why IDs are not true classes.'),('Reduce dimensions','30','Use digits. Compare classifier performance with and without PCA fitted within the evaluation pipeline.'),('Multiple labels','31','Generate a small multilabel task. Explain micro F1 and subset accuracy with one example of each.'),('Respect groups','23','Generate repeated group IDs for a small classification dataset. Demonstrate group-disjoint folds and explain the intended deployment.'),('A usable handover','34–35','Save an earlier pipeline. Write a script accepting a documented input and rejecting two invalid cases.'),('Your own bounded problem','32–35','Choose a new small dataset. Build one baseline and one candidate, report held-out evidence and limitations, and provide a prediction handover.')]

def line_note(line):
    if '#' in line: return line.split('#',1)[1].strip()
    s=line.strip()
    if s.startswith('from '): return 'Import the named tools from this package module so they can be used below.'
    if s.startswith('import '): return 'Load the package; an alias after as provides a shorter name.'
    if s.startswith('def '): return 'Define a reusable function; the indented lines run when it is called.'
    if s.startswith('class '): return 'Define a class describing a reusable object or request schema.'
    if s.startswith('for '): return 'Repeat the indented statements for each value in this sequence.'
    if s.startswith('if '): return 'Run the indented block only when this condition is true.'
    if s.startswith('try:'): return 'Start a block whose expected failure will be handled below.'
    if s.startswith('except '): return 'Handle the specified error rather than crashing the demonstration.'
    if s.startswith('@'): return 'Register the following function as a handler for this API route.'
    return 'Execute this statement in sequence; inspect the named objects and the explanation above.'

def build():
    import sklearn,numpy,pandas,scipy,matplotlib,seaborn,joblib
    out=ROOT/'dist'; (out/'notebooks').mkdir(parents=True,exist_ok=True)
    (out/'downloads').mkdir(exist_ok=True)
    output_records=[]
    for l in LESSONS:
        l['notes']=[{'line':i+1,'code':line,'note':line_note(line)} for i,line in enumerate(l['code'].splitlines()) if line.strip()]
        if l['code']:
            with tempfile.TemporaryDirectory() as td:
                p=subprocess.run([sys.executable,'-c',l['code']],cwd=td,capture_output=True,text=True,env={**os.environ,'MPLBACKEND':'Agg','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},timeout=90)
                if p.returncode: raise RuntimeError(f"Chapter {l['id']} failed: {p.stderr}")
                l['output']=p.stdout.strip()
                l['warnings']=p.stderr.strip()
                output_records.append({'chapter':l['id'],'passed':True,'output':l['output'],'warnings':l['warnings']})
                for img in Path(td).glob('*.png'):
                    (out/'images').mkdir(exist_ok=True)
                    import shutil
                    shutil.copy(img,out/'images'/f"chapter-{l['id']}.png")
                    l['image']=f"images/chapter-{l['id']}.png"
        else: l['output']=''; l['warnings']=''
        cells=[{'cell_type':'markdown','metadata':{},'source':[f"# {l['id']}. {l['title']}\n",l['idea'],'\n\n',l['walk']]}]
        if l['code']: cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':l['code'].splitlines(keepends=True)})
        cells.append({'cell_type':'markdown','metadata':{},'source':['## Your turn\n',l['task'],'\n\nTry the task before opening the solution on the course website.']})
        notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}},'nbformat':4,'nbformat_minor':5}
        (out/'notebooks'/f"chapter-{l['id']:02}.ipynb").write_text(json.dumps(notebook,indent=2))
    modules=[]
    for i,m in enumerate(MODULES):
        project=PROJECTS[i]
        modules.append(dict(id=i+1,title=m[0],description=m[1],start=m[2],end=m[3],hours=m[4],optional=i==10,project=dict(title=project[0],dataset=project[1],url=project[2],access=project[3],deliverables=project[4],plan=project[5])))
    # The first twenty questions are selected from chapter-specific items, not repeated placeholder questions.
    for module in modules:
        bank=[]
        for l in LESSONS:
            if l['module']!=module['id']:continue
            for j,t in enumerate(l['terms']):
                distractors=[x['meaning'] for x in l['terms'] if x!=t]
                bank.append(dict(id=f"{l['id']}-term-{j}",chapter=l['id'],type='Knowledge',prompt=f'What does “{t["term"]}” mean in this lesson?',options=[t['meaning']]+distractors,correct=0,explanation=t['meaning']))
            if l['code']:
                bank.append(dict(id=f"{l['id']}-output",chapter=l['id'],type='Predict the output',prompt=f"After running the complete Chapter {l['id']} example in the frozen environment, which output is expected?",options=[l['output'],'No output: none of the print statements run.','A single probability is returned for every model and every task.'],correct=0,explanation=l['reading'],code=l['code']))
                bank.append(dict(id=f"{l['id']}-first-output",chapter=l['id'],type='Predict the output',prompt='What is the first line printed by this complete example?',options=[l['output'].splitlines()[0], 'None', 'The code has no print statement.'],correct=0,explanation=l['reading'],code=l['code']))
            bank.append(dict(id=f"{l['id']}-reason",chapter=l['id'],type='Written explanation',prompt=l['question'],options=[],correct=0,explanation=l['answer']))
            bank.append(dict(id=f"{l['id']}-debug",chapter=l['id'],type='Debugging',prompt=f'In Chapter {l["id"]}, how should you address this problem: {l["error"]}?',options=[l['fix'],'Choose settings based on the final test results until the error disappears.','Suppress the problem and report the original output as correct.'],correct=0,explanation=l['fix']))
        # Add contextual review questions to smaller modules to support twenty-question tests.
        extras=[('Which statement makes evaluation honest?','Use an appropriate held-out partition and keep development choices separate from final testing.','Choose every setting from the final-test score.'),('What makes a result reproducible?','Record the data instructions, environment, execution steps, and relevant random seeds.','Keep only a screenshot of the highest score.'),('What should an explanation say about a number?','Name its units or meaning, evaluation partition, and relevant limitations.','Report it without context because larger numbers are always better.'),('What is the purpose of a baseline?','Provide a simple reference appropriate to the task.','Guarantee that a complex model will be useful.')]
        for j,(prompt,a,b) in enumerate(extras):bank.append(dict(id=f"m{module['id']}-review-{j}",chapter=module['start'],type='Debugging',prompt=prompt,options=[a,b],correct=0,explanation=a))
        module['questions']=bank
    env={'Python':sys.version.split()[0],'scikit-learn':sklearn.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'seaborn':seaborn.__version__,'joblib':joblib.__version__}
    data=dict(title='Scikit-learn Studio',modules=modules,lessons=LESSONS,challenge=[dict(day=i+1,title=x[0],chapters=x[1],brief=x[2]) for i,x in enumerate(CHALLENGE)],environment=env)
    (out/'course.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    (ROOT/'content'/'execution-report.json').write_text(json.dumps(dict(environment=env,examples=output_records),indent=2))
    req='\n'.join(f'{k}=={v}' for k,v in env.items() if k!='Python')+'\n'
    (ROOT/'requirements.txt').write_text(req)
    (out/'downloads'/'requirements.txt').write_text(req)
    optional='imbalanced-learn==0.14.1\nxgboost-cpu==3.1.2\nfastapi=='+__import__('fastapi').__version__+'\nhttpx=='+__import__('httpx').__version__+'\nsklearn-compat==0.1.6\nuvicorn=='+__import__('uvicorn').__version__+'\npydantic=='+__import__('pydantic').__version__+'\nstarlette=='+__import__('starlette').__version__+'\n'
    (ROOT/'requirements-optional.txt').write_text(optional)
    (out/'downloads'/'requirements-optional.txt').write_text(optional)
    flawed=CLASS+'''\n# EXERCISE: This notebook deliberately contains evaluation mistakes. Do not report these scores as valid.\nscaler = StandardScaler()\nZ = scaler.fit_transform(X)\nfrom sklearn.model_selection import train_test_split\nZ_train, Z_test, y_train, y_test = train_test_split(Z, y, random_state=42)\nfor c in [0.01, 1, 100]:\n    candidate = LogisticRegression(C=c, max_iter=1000).fit(Z_train, y_train)\n    print(c, candidate.score(Z_test, y_test))\nprint("Choose the best test score and claim this is guaranteed future accuracy.")\n'''
    (out/'downloads'/'flawed-workflow.py').write_text(flawed)
    # A second assessment includes a wrong split, misleading metric, feature leakage, and an unsupported claim.
    (out/'downloads'/'diagnosis-mission.py').write_text('''# Deliberately flawed. Diagnose each mistake; do not use as a valid workflow.\nimport numpy as np\nfrom sklearn.datasets import make_classification\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nX, y = make_classification(n_samples=200, n_features=5, weights=[0.98, 0.02], random_state=42)\ngroups = np.repeat(np.arange(50), 4)\nX = np.column_stack([X, y])  # An unavailable outcome column!\nZ = StandardScaler().fit_transform(X)\nXt, Xv, yt, yv = train_test_split(Z, y, random_state=42)\nfor c in [0.1, 1, 10]:\n    model = LogisticRegression(C=c).fit(Xt, yt)\n    print("Final test accuracy", model.score(Xv, yv))\nprint("This proves the model will work perfectly for unseen groups.")\n''')
    import shutil, zipfile
    outline='# Scikit-learn: From Foundations to Independent Projects\n\nCourse outline and lesson index.\n\n'
    for m in modules:
        outline+=f"## Module {m['id']}: {m['title']}\n\n{m['description']}\n\n"
        for lesson in LESSONS:
            if lesson['module']==m['id']:outline+=f"- Chapter {lesson['id']}: {lesson['title']}\n"
        outline+='\n'
    (out/'downloads'/'course-outline.md').write_text(outline)
    shutil.copy(ROOT/'content'/'execution-report.json',out/'downloads'/'execution-report.json')
    with zipfile.ZipFile(out/'downloads'/'practice-notebooks.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (out/'notebooks').glob('*.ipynb'):z.write(p,'notebooks/'+p.name)
        z.write(ROOT/'requirements.txt','requirements.txt')
        z.write(ROOT/'requirements-optional.txt','requirements-optional.txt')
    print(f'Built {len(LESSONS)} lesson pages; executed {len(output_records)} standalone examples; generated {sum(len(m["questions"]) for m in modules)} question-bank items.')
if __name__=='__main__':build()
