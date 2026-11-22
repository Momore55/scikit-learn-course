from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_breast_cancer(return_X_y=True)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# EXERCISE: This notebook deliberately contains evaluation mistakes. Do not report these scores as valid.
scaler = StandardScaler()
Z = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
Z_train, Z_test, y_train, y_test = train_test_split(Z, y, random_state=42)
for c in [0.01, 1, 100]:
    candidate = LogisticRegression(C=c, max_iter=1000).fit(Z_train, y_train)
    print(c, candidate.score(Z_test, y_test))
print("Choose the best test score and claim this is guaranteed future accuracy.")
