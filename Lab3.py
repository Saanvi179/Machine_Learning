# ============================================
# Imports (ONCE)
# ============================================
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# Load Dataset (ONCE)
# ============================================
df = pd.read_csv(r'D:\Engineering\4th Semester\Machine Learning\data.csv')

# ============================================
# A1: Dot Product and Euclidean Norm
# ============================================

def dot_product(A, B):
    if len(A) != len(B):
        raise ValueError("Vectors must be of same length")
    result = 0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

def euclidean_norm(A):
    sum_sq = 0
    for val in A:
        sum_sq += val ** 2
    return math.sqrt(sum_sq)

A = np.array([2, 3, 4])
B = np.array([1, 0, 6])

print("Dot Product (My Function):", dot_product(A, B))
print("Dot Product (NumPy):", np.dot(A, B))
print("Euclidean Norm (My Function):", euclidean_norm(A))
print("Euclidean Norm (NumPy):", np.linalg.norm(A))

# ============================================
# A2: Intraclass Spread & Interclass Distance
# ============================================

def own_mean(x):
    return sum(x) / len(x)

def own_var(x):
    me = own_mean(x)
    var = 0
    for i in range(len(x)):
        var += (x[i] - me) ** 2
    return var / (len(x) - 1)

# Extract class labels from filename
df["class"] = df["filename"].str.extract(r"(\d+)")

class1 = "172"
class2 = "173"

dc1 = df[df["class"] == class1]
dc2 = df[df["class"] == class2]

features = df.columns[df.columns.str.startswith("MFCC")]

X1 = dc1[features].values
X2 = dc2[features].values

print("Mean of first feature:",
      own_mean(X1[:, 0]),
      own_mean(X2[:, 0]))

centroid1 = X1.mean(axis=0)
centroid2 = X2.mean(axis=0)

interclass_dist = np.linalg.norm(centroid1 - centroid2)
print("Interclass Distance:", interclass_dist)

avg_spread1 = X1.std(axis=0).mean()
avg_spread2 = X2.std(axis=0).mean()

print("Intraclass Spread:", avg_spread1, avg_spread2)

# ============================================
# A3: Histogram Visualization
# ============================================

plt.figure()
plt.hist(df["MFCC_0_mean"], bins=20)
plt.xlabel("MFCC_0 values")
plt.ylabel("Frequency")
plt.title("Histogram of MFCC_0")
plt.show()
plt.close()

# ============================================
# A4: Minkowski Distance (p = 1 to 10)
# ============================================

features = df.columns[df.columns.str.startswith("MFCC")]
v1 = df[features].iloc[0].values
v2 = df[features].iloc[1].values

def minkowski_distance(x, y, p):
    dist = 0
    for i in range(len(x)):
        dist += abs(x[i] - y[i]) ** p
    return dist ** (1 / p)

p_values = list(range(1, 11))
distances = [minkowski_distance(v1, v2, p) for p in p_values]

plt.figure()
plt.plot(p_values, distances, marker='o')
plt.xlabel("p value")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance vs p")
plt.grid(True)
plt.show()
plt.close()

# ============================================
# A5: Own Minkowski vs SciPy Minkowski
# ============================================

from scipy.spatial.distance import minkowski

p = 2  # Euclidean distance case

own_dist = minkowski_distance(v1, v2, p)
scipy_dist = minkowski(v1, v2, p)

print("Own Minkowski Distance (p=2):", own_dist)
print("SciPy Minkowski Distance (p=2):", scipy_dist)

# ============================================
# A6: Train-Test Split
# ============================================

from sklearn.model_selection import train_test_split

# Use same MFCC features
features = df.columns[df.columns.str.startswith("MFCC")]
X = df[features].values

# Use class labels already created in A2
y = df["class"].values

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


# ============================================
# A7: Train kNN Classifier (k = 3)
# ============================================

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ============================================
# A8: Test Accuracy
# ============================================

accuracy = knn.score(X_test, y_test)
print("kNN Accuracy (k=3):", accuracy)

# ============================================
# A9: Prediction Behavior
# ============================================

predictions = knn.predict(X_test)
print("Predictions on test set:", predictions[:10])

# Predict for a single test vector
single_prediction = knn.predict([X_test[0]])
print("Prediction for one test vector:", single_prediction)

from collections import Counter
import numpy as np

# -----------------------------
# A 10: Own kNN Implementation
# -----------------------------

# =============================
# Own Metric Functions (SAFE)
# =============================

def own_knn_predict(X_train, y_train, x_test, k=3):
    distances = []
    for i in range(len(X_train)):
        dist = np.linalg.norm(X_train[i] - x_test)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda tup: tup[0])
    k_nearest_labels = [label for (_, label) in distances[:k]]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

def own_accuracy(TP, FP, FN, TN):
    total = TP + FP + FN + TN
    return (TP + TN) / total if total != 0 else 0

def own_precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def own_recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def own_f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0


# -----------------------------
# Predict using Own kNN
# -----------------------------

own_predictions = []
for x in X_test:
    own_predictions.append(own_knn_predict(X_train, y_train, x, k=3))

own_predictions = np.array(own_predictions)

# Accuracy of own kNN
own_knn_accuracy = np.mean(own_predictions == y_test)

print("Own kNN Accuracy (k=3):", own_knn_accuracy)


# ============================================
# A11: Accuracy vs k (k = 1 to 11)
# ============================================

from sklearn.neighbors import KNeighborsClassifier

k_values = list(range(1, 12))
accuracies = []

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    acc = knn_k.score(X_test, y_test)
    accuracies.append(acc)

# Plot accuracy vs k
plt.figure()
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k in kNN")
plt.grid(True)
plt.show()

print("Accuracy for k=1:", accuracies[0])
print("Accuracy for k=3:", accuracies[2])


# ============================================
# A12: Confusion Matrix & Metrics
# ============================================

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Predictions
train_pred = knn.predict(X_train)
test_pred = knn.predict(X_test)

# Confusion matrices
cm_train = confusion_matrix(y_train, train_pred)
cm_test = confusion_matrix(y_test, test_pred)

print("Confusion Matrix (Train):\n", cm_train)
print("Confusion Matrix (Test):\n", cm_test)

# Metrics - Train
print("Train Precision:", precision_score(y_train, train_pred, average='macro'))
print("Train Recall:", recall_score(y_train, train_pred, average='macro'))
print("Train F1 Score:", f1_score(y_train, train_pred, average='macro'))

# Metrics - Test
print("Test Precision:", precision_score(y_test, test_pred, average='macro'))
print("Test Recall:", recall_score(y_test, test_pred, average='macro'))
print("Test F1 Score:", f1_score(y_test, test_pred, average='macro'))

# ============================================
# A13: Own Metrics Functions
# ============================================

def own_confusion_matrix(y_true, y_pred):
    TP = FP = FN = TN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
    return TP, FP, FN, TN

def own_accuracy(TP, FP, FN, TN):
    return (TP + TN) / (TP + FP + FN + TN)

def own_precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def own_recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def own_f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

TP, FP, FN, TN = own_confusion_matrix(y_test, test_pred)

print("Own Accuracy:", own_accuracy(TP, FP, FN, TN))
print("Own Precision:", own_precision(TP, FP))
print("Own Recall:", own_recall(TP, FN))
print("Own F1 Score:", own_f1(own_precision(TP, FP), own_recall(TP, FN)))

# ============================================
# A14: Matrix Inversion Technique
# ============================================

# Add bias term
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Closed-form solution
W = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

# Predictions
y_pred_linear = (X_test_b @ W >= 0.5).astype(int)

# Accuracy
linear_accuracy = np.mean(y_pred_linear == y_test)

print("kNN Accuracy:", accuracy)
print("Matrix Inversion Accuracy:", linear_accuracy)
