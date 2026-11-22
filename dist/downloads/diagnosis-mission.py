# Deliberately flawed. Diagnose each mistake; do not use as a valid workflow.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X, y = make_classification(n_samples=200, n_features=5, weights=[0.98, 0.02], random_state=42)
groups = np.repeat(np.arange(50), 4)
X = np.column_stack([X, y])  # An unavailable outcome column!
Z = StandardScaler().fit_transform(X)
Xt, Xv, yt, yv = train_test_split(Z, y, random_state=42)
for c in [0.1, 1, 10]:
    model = LogisticRegression(C=c).fit(Xt, yt)
    print("Final test accuracy", model.score(Xv, yv))
print("This proves the model will work perfectly for unseen groups.")
