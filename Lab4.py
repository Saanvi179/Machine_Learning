import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                             precision_score, recall_score, f1_score)

# -------------------------------------------------
# LOAD AND PREPROCESS DATA
# -------------------------------------------------

df = pd.read_csv(r'D:\Engineering\4th Semester\Machine Learning\data.csv')

df = df.drop(columns=["filename"])
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.mean())
df = df.fillna(0)

X = df.drop(columns=["Fluency"])
y = df["Fluency"]

valid_idx = ~y.isna()
X = X[valid_idx]
y = y[valid_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# TRAIN KNN MODEL
# -------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# -------------------------------------------------
# PREDICTIONS
# -------------------------------------------------

train_pred = knn.predict(X_train)
test_pred = knn.predict(X_test)

# -------------------------------------------------
# CONFUSION MATRICES
# -------------------------------------------------

cm_train = confusion_matrix(y_train, train_pred)
cm_test = confusion_matrix(y_test, test_pred)

print("="*80)
print("A1: CONFUSION MATRIX & PERFORMANCE METRICS EVALUATION")
print("="*80)

print("\n" + "="*80)
print("CONFUSION MATRICES")
print("="*80)

print("\n===== TRAIN CONFUSION MATRIX =====")
print("(Rows: Actual, Columns: Predicted)")
print(cm_train)

print("\n===== TEST CONFUSION MATRIX =====")
print("(Rows: Actual, Columns: Predicted)")
print(cm_test)

# -------------------------------------------------
# EXTRACT CONFUSION MATRIX COMPONENTS
# -------------------------------------------------

# For binary classification (assuming 2 classes)
if cm_train.shape[0] == 2:
    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()
    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX COMPONENTS BREAKDOWN")
    print("="*80)
    
    print("\nTRAIN SET:")
    print(f"  True Negatives (TN):  {tn_train}")
    print(f"  False Positives (FP): {fp_train}")
    print(f"  False Negatives (FN): {fn_train}")
    print(f"  True Positives (TP):  {tp_train}")
    
    print("\nTEST SET:")
    print(f"  True Negatives (TN):  {tn_test}")
    print(f"  False Positives (FP): {fp_test}")
    print(f"  False Negatives (FN): {fn_test}")
    print(f"  True Positives (TP):  {tp_test}")

# -------------------------------------------------
# PERFORMANCE METRICS (Weighted Average)
# -------------------------------------------------

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return acc, prec, rec, f1

train_acc, train_prec, train_rec, train_f1 = get_metrics(y_train, train_pred)
test_acc, test_prec, test_rec, test_f1 = get_metrics(y_test, test_pred)

print("\n" + "="*80)
print("WEIGHTED AVERAGE METRICS")
print("="*80)

print("\n===== TRAIN METRICS (Weighted Average) =====")
print(f"Accuracy:  {train_acc:.4f}")
print(f"Precision: {train_prec:.4f}")
print(f"Recall:    {train_rec:.4f}")
print(f"F1 Score:  {train_f1:.4f}")

print("\n===== TEST METRICS (Weighted Average) =====")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1 Score:  {test_f1:.4f}")

# -------------------------------------------------
# PER-CLASS METRICS (Classification Report)
# -------------------------------------------------

print("\n" + "="*80)
print("PER-CLASS METRICS (Classification Report)")
print("="*80)

print("\n===== TRAIN SET CLASSIFICATION REPORT =====")
print(classification_report(y_train, train_pred, digits=4))

print("\n===== TEST SET CLASSIFICATION REPORT =====")
print(classification_report(y_test, test_pred, digits=4))

# -------------------------------------------------
# MACRO & MICRO AVERAGES
# -------------------------------------------------

train_prec_macro = precision_score(y_train, train_pred, average="macro")
test_prec_macro = precision_score(y_test, test_pred, average="macro")

train_rec_macro = recall_score(y_train, train_pred, average="macro")
test_rec_macro = recall_score(y_test, test_pred, average="macro")

train_f1_macro = f1_score(y_train, train_pred, average="macro")
test_f1_macro = f1_score(y_test, test_pred, average="macro")

print("\n" + "="*80)
print("MACRO AVERAGE METRICS")
print("="*80)

print("\n===== TRAIN METRICS (Macro Average) =====")
print(f"Precision (Macro): {train_prec_macro:.4f}")
print(f"Recall (Macro):    {train_rec_macro:.4f}")
print(f"F1 Score (Macro):  {train_f1_macro:.4f}")

print("\n===== TEST METRICS (Macro Average) =====")
print(f"Precision (Macro): {test_prec_macro:.4f}")
print(f"Recall (Macro):    {test_rec_macro:.4f}")
print(f"F1 Score (Macro):  {test_f1_macro:.4f}")

# -------------------------------------------------
# CONFUSION MATRIX VISUALIZATION
# -------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train Confusion Matrix Heatmap
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Train Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=12)

# Test Confusion Matrix Heatmap
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'],
            cbar_kws={'label': 'Count'})
axes[1].set_title('Test Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_xlabel('Predicted', fontsize=12)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# METRICS COMPARISON (Train vs Test)
# -------------------------------------------------

metrics_comparison = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Train': [train_acc, train_prec, train_rec, train_f1],
    'Test': [test_acc, test_prec, test_rec, test_f1]
}

comparison_df = pd.DataFrame(metrics_comparison)
comparison_df['Difference'] = comparison_df['Train'] - comparison_df['Test']

print("\n" + "="*80)
print("TRAIN vs TEST METRICS COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metrics_comparison['Metric']))
width = 0.35

ax.bar(x - width/2, metrics_comparison['Train'], width, label='Train', alpha=0.8, color='skyblue')
ax.bar(x + width/2, metrics_comparison['Test'], width, label='Test', alpha=0.8, color='orange')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Train vs Test Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_comparison['Metric'])
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

for i, (train_val, test_val) in enumerate(zip(metrics_comparison['Train'], metrics_comparison['Test'])):
    ax.text(i - width/2, train_val + 0.02, f'{train_val:.3f}', ha='center', fontsize=10)
    ax.text(i + width/2, test_val + 0.02, f'{test_val:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# MODEL FIT DIAGNOSIS (Underfitting/Overfitting)
# -------------------------------------------------

print("\n" + "="*80)
print("MODEL FIT DIAGNOSIS - LEARNING OUTCOME ANALYSIS")
print("="*80)

accuracy_gap = train_acc - test_acc
precision_gap = train_prec - test_prec
recall_gap = train_rec - test_rec
f1_gap = train_f1 - test_f1

print(f"\nMetric Gaps (Train - Test):")
print(f"  Accuracy Gap:  {accuracy_gap:.4f}")
print(f"  Precision Gap: {precision_gap:.4f}")
print(f"  Recall Gap:    {recall_gap:.4f}")
print(f"  F1 Score Gap:  {f1_gap:.4f}")

print("\n" + "-"*80)

if train_acc > 0.90 and test_acc < 0.70:
    print("DIAGNOSIS: OVERFITTING")
    print("\nExplanation:")
    print(f"  - Train Accuracy: {train_acc:.4f} (HIGH)")
    print(f"  - Test Accuracy:  {test_acc:.4f} (LOW)")
    print(f"  - Accuracy Gap:   {accuracy_gap:.4f} (LARGE GAP > 0.20)")
    print("\nThe model performs much better on training data than test data.")
    print("This indicates the model has memorized training patterns and")
    print("fails to generalize to unseen data.")
    print("\nRecommendations:")
    print("  • Increase K value (use K=5, K=7, etc.)")
    print("  • Collect more training data")
    print("  • Use regularization techniques")
    print("  • Perform feature selection")

elif train_acc < 0.70 and test_acc < 0.70:
    print("DIAGNOSIS: UNDERFITTING")
    print("\nExplanation:")
    print(f"  - Train Accuracy: {train_acc:.4f} (LOW)")
    print(f"  - Test Accuracy:  {test_acc:.4f} (LOW)")
    print("\nThe model performs poorly on both training and test data.")
    print("This indicates the model is too simple to capture data patterns.")
    print("\nRecommendations:")
    print("  • Decrease K value (use K=1, K=2, etc.)")
    print("  • Add more features or engineer new features")
    print("  • Use a more complex model")
    print("  • Check data quality and preprocessing")

elif abs(accuracy_gap) < 0.10:
    print("DIAGNOSIS: WELL-FIT (REGULAR FIT)")
    print("\nExplanation:")
    print(f"  - Train Accuracy: {train_acc:.4f}")
    print(f"  - Test Accuracy:  {test_acc:.4f}")
    print(f"  - Accuracy Gap:   {accuracy_gap:.4f} (SMALL GAP < 0.10)")
    print("\nThe model performs similarly on both training and test data.")
    print("This indicates good generalization and balanced learning.")
    print("\nThe model is neither overfitting nor underfitting.")

else:
    print("DIAGNOSIS: INTERMEDIATE CASE")
    print("\nExplanation:")
    print(f"  - Train Accuracy: {train_acc:.4f}")
    print(f"  - Test Accuracy:  {test_acc:.4f}")
    print(f"  - Accuracy Gap:   {accuracy_gap:.4f}")
    print("\nReview specific metrics and confusion matrix for detailed analysis.")

print("\n" + "="*80)

# -------------------------------------------------
# A2
# -------------------------------------------------

import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank


df = pd.read_excel(
    r"d:\Engineering\4th Semester\Machine Learning\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = df["Payment (Rs)"].values.reshape(-1, 1)

print("Feature matrix shape:", X.shape)
print("Output vector shape:", y.shape)

X_pinv = np.linalg.pinv(X)
w = X_pinv @ y

print("\nLearned Product Prices:")
print("Candy Price  :", w[0, 0])
print("Mango Price  :", w[1, 0])
print("Milk Price   :", w[2, 0])

y_pred = X @ w

print("\n" + "="*60)
print("ACTUAL vs PREDICTED PAYMENT")
print("="*60)
print("\nActual Payments:")
print(y.flatten())
print("\nPredicted Payments:")
print(y_pred.flatten())


y_true = y.flatten()
y_pred_flat = y_pred.flatten()

mse = np.mean((y_true - y_pred_flat) ** 2)

rmse = np.sqrt(mse)

mape = np.mean(np.abs((y_true - y_pred_flat) / y_true)) * 100

ss_res = np.sum((y_true - y_pred_flat) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_score = 1 - (ss_res / ss_tot)


print("\n" + "="*60)
print("REGRESSION METRICS")
print("="*60)

print(f"\n1. Mean Squared Error (MSE)      : {mse:.4f}")
print(f"2. Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"3. Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
print(f"4. R² Score (Coefficient of Determination): {r2_score:.4f}")


errors = y_true - y_pred_flat
abs_errors = np.abs(errors)
percent_errors = np.abs(errors / y_true) * 100

print("\n" + "="*60)
print("DETAILED ERROR ANALYSIS")
print("="*60)

print("\nError Statistics:")
print(f"Mean Error           : {np.mean(errors):.4f}")
print(f"Std Dev of Errors    : {np.std(errors):.4f}")
print(f"Min Error            : {np.min(errors):.4f}")
print(f"Max Error            : {np.max(errors):.4f}")
print(f"Mean Absolute Error  : {np.mean(abs_errors):.4f}")


results_df = pd.DataFrame({
    "Candies": X[:, 0],
    "Mangoes": X[:, 1],
    "Milk": X[:, 2],
    "Actual_Payment": y_true,
    "Predicted_Payment": y_pred_flat,
    "Error": errors,
    "Abs_Error": abs_errors,
    "Percent_Error_%": percent_errors
})

print("\n" + "="*60)
print("PREDICTION DETAILS FOR ALL SAMPLES")
print("="*60)
print(results_df.to_string())


print("\n" + "="*60)
print("ANALYSIS & INTERPRETATION")
print("="*60)

print("\n1. MSE (Mean Squared Error) = {:.4f}".format(mse))
print("   - Measures average squared difference between actual and predicted")
print("   - Lower MSE indicates better fit")
print("   - Units: Rs²")

print("\n2. RMSE (Root Mean Squared Error) = {:.4f}".format(rmse))
print("   - Square root of MSE")
print("   - Same units as target variable (Rs)")
print("   - Easier to interpret: On average, predictions are off by ±{:.2f} Rs".format(rmse))

print("\n3. MAPE (Mean Absolute Percentage Error) = {:.4f}%".format(mape))
print("   - Average percentage deviation from actual values")
if mape < 5:
    print("   - EXCELLENT: Very accurate predictions (< 5%)")
elif mape < 10:
    print("   - GOOD: Accurate predictions (5-10%)")
elif mape < 15:
    print("   - ACCEPTABLE: Reasonable predictions (10-15%)")
else:
    print("   - POOR: Predictions need improvement (> 15%)")

print("\n4. R² Score = {:.4f}".format(r2_score))
print("   - Measures proportion of variance explained by the model")
print("   - Range: 0 to 1 (higher is better)")
if r2_score > 0.9:
    print("   - EXCELLENT: Model explains {:.2f}% of variance".format(r2_score * 100))
elif r2_score > 0.7:
    print("   - GOOD: Model explains {:.2f}% of variance".format(r2_score * 100))
elif r2_score > 0.5:
    print("   - ACCEPTABLE: Model explains {:.2f}% of variance".format(r2_score * 100))
else:
    print("   - POOR: Model explains {:.2f}% of variance".format(r2_score * 100))

print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)

if mape < 10 and r2_score > 0.8:
    print("\n✓ GOOD MODEL: Both MAPE and R² indicate good predictions")
elif mape < 15 and r2_score > 0.6:
    print("\n~ ACCEPTABLE MODEL: Reasonable predictions with some errors")
else:
    print("\n✗ POOR MODEL: High errors, needs improvement or more data")

print("\nPossible Improvements:")
print("- Collect more training data")
print("- Check for outliers in the data")
print("- Consider non-linear relationships")
print("- Verify data quality and measurement accuracy")

# =====================================================
# A3: KNN CLASSIFICATION WITH 20 TRAINING DATA POINTS
# =====================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

X_train = np.random.uniform(1, 10, size=(20, 2))

y_train = np.where((X_train[:, 0] + X_train[:, 1]) > 11, 1, 0)

class_0_count = np.sum(y_train == 0)
class_1_count = np.sum(y_train == 1)

print("Training Data Points with Class Labels")
print("-" * 70)
print(f"{'#':>3} {'X':>8} {'Y':>8} {'X+Y':>8} {'Class':>8} {'Color':>10}")
print("-" * 70)

for i in range(20):
    x_val = X_train[i, 0]
    y_val = X_train[i, 1]
    sum_val = x_val + y_val
    class_label = y_train[i]
    color = "Red" if class_label == 1 else "Blue"
    print(f"{i+1:>3} {x_val:>8.2f} {y_val:>8.2f} {sum_val:>8.2f} {class_label:>8} {color:>10}")

plt.figure(figsize=(10, 8))

class_0_points = X_train[y_train == 0]
class_1_points = X_train[y_train == 1]

plt.scatter(class_0_points[:, 0], class_0_points[:, 1], 
            c='blue', s=150, marker='o', label='Class 0 (Blue)', 
            edgecolors='black', linewidth=1.5, alpha=0.7)

plt.scatter(class_1_points[:, 0], class_1_points[:, 1], 
            c='red', s=150, marker='o', label='Class 1 (Red)', 
            edgecolors='black', linewidth=1.5, alpha=0.7)

x_boundary = np.linspace(0, 12, 100)
y_boundary = 11 - x_boundary
plt.plot(x_boundary, y_boundary, 'g--', linewidth=2, label='Decision Boundary (X+Y=11)')

plt.xlabel('Feature X', fontsize=12, fontweight='bold')
plt.ylabel('Feature Y', fontsize=12, fontweight='bold')
plt.title('Training Data: 20 Points with 2 Classes', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.tight_layout()
plt.show()

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

h = 0.2
x_min, x_max = 0, 12
y_min, y_max = 0, 12

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 9))

plt.contourf(xx, yy, Z, alpha=0.3, levels=[0, 0.5, 1], colors=['lightblue', 'lightcoral'])
plt.contour(xx, yy, Z, colors='green', linewidths=2, levels=[0.5])

plt.scatter(class_0_points[:, 0], class_0_points[:, 1], 
            c='blue', s=150, marker='o', label='Class 0 (Blue)', 
            edgecolors='black', linewidth=1.5, zorder=5)

plt.scatter(class_1_points[:, 0], class_1_points[:, 1], 
            c='red', s=150, marker='o', label='Class 1 (Red)', 
            edgecolors='black', linewidth=1.5, zorder=5)

plt.xlabel('Feature X', fontsize=12, fontweight='bold')
plt.ylabel('Feature Y', fontsize=12, fontweight='bold')
plt.title(f'KNN Decision Boundary (K={k})', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.tight_layout()
plt.show()

test_points = np.array([
    [2, 3],
    [9, 8],
    [5, 5],
    [6, 6],
    [3, 10],
])

predictions = knn.predict(test_points)

print("\n" + "-" * 70)
print("Test Points Predictions:")
print("-" * 70)
print(f"{'Point #':>8} {'X':>8} {'Y':>8} {'Predicted Class':>20}")
print("-" * 70)

for i, (point, pred) in enumerate(zip(test_points, predictions)):
    class_name = "Class 1 (Red)" if pred == 1 else "Class 0 (Blue)"
    print(f"{i+1:>8} {point[0]:>8.1f} {point[1]:>8.1f} {class_name:>20}")

plt.figure(figsize=(12, 9))

plt.contourf(xx, yy, Z, alpha=0.25, levels=[0, 0.5, 1], colors=['lightblue', 'lightcoral'])
plt.contour(xx, yy, Z, colors='green', linewidths=2, levels=[0.5])

plt.scatter(class_0_points[:, 0], class_0_points[:, 1], 
            c='blue', s=150, marker='o', label='Training: Class 0 (Blue)', 
            edgecolors='black', linewidth=1.5, zorder=5)

plt.scatter(class_1_points[:, 0], class_1_points[:, 1], 
            c='red', s=150, marker='o', label='Training: Class 1 (Red)', 
            edgecolors='black', linewidth=1.5, zorder=5)

test_colors = ['blue' if p == 0 else 'red' for p in predictions]
plt.scatter(test_points[:, 0], test_points[:, 1], 
            c=test_colors, s=300, marker='*', label='Test Points', 
            edgecolors='black', linewidth=2, zorder=6)

plt.xlabel('Feature X', fontsize=12, fontweight='bold')
plt.ylabel('Feature Y', fontsize=12, fontweight='bold')
plt.title(f'KNN Classification (K={k}) - Training & Test Data', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.tight_layout()
plt.show()

train_predictions = knn.predict(X_train)
train_accuracy = np.mean(train_predictions == y_train)

print(f"\nTraining Data Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# =====================================================
# A4: KNN DECISION BOUNDARY VISUALIZATION
# WITH 10,000 TEST POINTS
# =====================================================

print("="*80)
print("A4: KNN DECISION BOUNDARY WITH DENSE TEST SET (10,000 Points)")
print("="*80)

np.random.seed(42)


X_train = np.random.uniform(1, 10, size=(20, 2))
y_train = np.where((X_train[:, 0] + X_train[:, 1]) > 11, 1, 0)

class_0_train = X_train[y_train == 0]
class_1_train = X_train[y_train == 1]

print("\nStep 1: Training Data")
print(f"  Total training points: 20")
print(f"  Class 0 (Blue): {len(class_0_train)} points")
print(f"  Class 1 (Red):  {len(class_1_train)} points")

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

print(f"\nStep 2: KNN Model Trained")
print(f"  K value: {k}")
print(f"  Total training samples: {len(X_train)}")


print("\nStep 3: Generating Dense Test Set")

X_test_range = np.arange(0, 10.1, 0.1)
Y_test_range = np.arange(0, 10.1, 0.1)

XX, YY = np.meshgrid(X_test_range, Y_test_range)

X_test = np.c_[XX.ravel(), YY.ravel()]

print(f"  X range: 0 to 10 with increment 0.1 → {len(X_test_range)} points")
print(f"  Y range: 0 to 10 with increment 0.1 → {len(Y_test_range)} points")
print(f"  Total test points: {len(X_test)} points")


print("\nStep 4: Classifying Test Points")

y_test_pred = knn.predict(X_test)

class_0_test = X_test[y_test_pred == 0]
class_1_test = X_test[y_test_pred == 1]

print(f"  Points predicted as Class 0 (Blue): {len(class_0_test)}")
print(f"  Points predicted as Class 1 (Red):  {len(class_1_test)}")

print("\nStep 5: Creating Visualizations")

plt.figure(figsize=(12, 10))

plt.scatter(class_0_test[:, 0], class_0_test[:, 1], 
            c='blue', s=20, alpha=0.6, label='Predicted Class 0 (Blue)')

plt.scatter(class_1_test[:, 0], class_1_test[:, 1], 
            c='red', s=20, alpha=0.6, label='Predicted Class 1 (Red)')

plt.scatter(class_0_train[:, 0], class_0_train[:, 1], 
            c='darkblue', s=150, marker='o', edgecolors='black', 
            linewidth=1.5, label='Training Class 0 (Dark Blue)', zorder=5)

plt.scatter(class_1_train[:, 0], class_1_train[:, 1], 
            c='darkred', s=150, marker='o', edgecolors='black', 
            linewidth=1.5, label='Training Class 1 (Dark Red)', zorder=5)

x_boundary = np.linspace(0, 10, 100)
y_boundary = 11 - x_boundary
plt.plot(x_boundary, y_boundary, 'g--', linewidth=3, label='True Boundary (X+Y=11)', zorder=4)

plt.xlabel('Feature X', fontsize=13, fontweight='bold')
plt.ylabel('Feature Y', fontsize=13, fontweight='bold')
plt.title(f'KNN Decision Boundary (K={k}) - 10,000 Test Points Classification', 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.tight_layout()
plt.show()


h = 0.05
x_min, x_max = 0, 10
y_min, y_max = 0, 10

xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h),
                     np.arange(y_min, y_max + h, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(14, 11))

plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightblue', 'lightcoral'], alpha=0.5)
plt.contour(xx, yy, Z, colors='green', linewidths=3, levels=[0.5], label='KNN Decision Boundary')

plt.scatter(class_0_test[:, 0], class_0_test[:, 1], 
            c='blue', s=15, alpha=0.5, label='Predicted Class 0')

plt.scatter(class_1_test[:, 0], class_1_test[:, 1], 
            c='red', s=15, alpha=0.5, label='Predicted Class 1')

plt.scatter(class_0_train[:, 0], class_0_train[:, 1], 
            c='darkblue', s=200, marker='o', edgecolors='black', 
            linewidth=2, label='Training Class 0', zorder=5)

plt.scatter(class_1_train[:, 0], class_1_train[:, 1], 
            c='darkred', s=200, marker='o', edgecolors='black', 
            linewidth=2, label='Training Class 1', zorder=5)

x_boundary = np.linspace(0, 10, 100)
y_boundary = 11 - x_boundary
plt.plot(x_boundary, y_boundary, 'g--', linewidth=4, label='True Boundary (X+Y=11)', zorder=4)

plt.xlabel('Feature X', fontsize=13, fontweight='bold')
plt.ylabel('Feature Y', fontsize=13, fontweight='bold')
plt.title(f'KNN Decision Boundary with Color Regions (K={k})', 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.2)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("OBSERVATIONS & ANALYSIS")
print("="*80)

true_boundary = (X_test[:, 0] + X_test[:, 1]) > 11
knn_pred = y_test_pred == 1
boundary_accuracy = np.mean(true_boundary == knn_pred)

print(f"\nBoundary Classification Accuracy: {boundary_accuracy:.4f} ({boundary_accuracy*100:.2f}%)")

print("\nColor Spread Analysis:")
print(f"\n  Blue Region (Class 0):")
print(f"    - Number of points: {len(class_0_test)}")
print(f"    - Percentage: {(len(class_0_test)/len(X_test))*100:.2f}%")
print(f"    - Typical X+Y range: {class_0_test[:, 0].mean() + class_0_test[:, 1].mean():.2f} (average)")

print(f"\n  Red Region (Class 1):")
print(f"    - Number of points: {len(class_1_test)}")
print(f"    - Percentage: {(len(class_1_test)/len(X_test))*100:.2f}%")
print(f"    - Typical X+Y range: {class_1_test[:, 0].mean() + class_1_test[:, 1].mean():.2f} (average)")

distances_to_boundary = np.abs((X_test[:, 0] + X_test[:, 1]) - 11)
boundary_region_indices = distances_to_boundary < 2  # within 2 units of boundary

print(f"\nBoundary Region Analysis (within ±2 units of X+Y=11):")
print(f"  - Points near boundary: {np.sum(boundary_region_indices)}")
print(f"  - Percentage of total: {(np.sum(boundary_region_indices)/len(X_test))*100:.2f}%")

near_boundary_class0 = np.sum((boundary_region_indices) & (y_test_pred == 0))
near_boundary_class1 = np.sum((boundary_region_indices) & (y_test_pred == 1))

print(f"  - Class 0 near boundary: {near_boundary_class0}")
print(f"  - Class 1 near boundary: {near_boundary_class1}")


# =====================================================
# A5: KNN DECISION BOUNDARY FOR VARYING K VALUES
# =====================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

print("="*80)
print("A5: KNN DECISION BOUNDARY ANALYSIS - VARYING K VALUES")
print("="*80)

np.random.seed(42)


X_train = np.random.uniform(1, 10, size=(20, 2))
y_train = np.where((X_train[:, 0] + X_train[:, 1]) > 11, 1, 0)

class_0_train = X_train[y_train == 0]
class_1_train = X_train[y_train == 1]

print("\nTraining Data:")
print(f"  Total points: 20")
print(f"  Class 0: {len(class_0_train)} points")
print(f"  Class 1: {len(class_1_train)} points")

print("\nTest Data Grid:")
X_range = np.arange(0, 10.1, 0.1)
Y_range = np.arange(0, 10.1, 0.1)
XX, YY = np.meshgrid(X_range, Y_range)
X_test = np.c_[XX.ravel(), YY.ravel()]

print(f"  Total test points: {len(X_test)}")

k_values = [1, 3, 5, 7, 10, 15]

print(f"\nK Values to Compare: {k_values}")

print("\nTraining KNN models for each K value...")

models_data = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)
    
    # Calculate boundary accuracy
    true_boundary = (X_test[:, 0] + X_test[:, 1]) > 11
    knn_pred = y_test_pred == 1
    boundary_accuracy = np.mean(true_boundary == knn_pred)
    
    # Count class distribution
    class_0_count = np.sum(y_test_pred == 0)
    class_1_count = np.sum(y_test_pred == 1)
    
    models_data[k] = {
        'model': knn,
        'predictions': y_test_pred,
        'class_0_count': class_0_count,
        'class_1_count': class_1_count,
        'accuracy': boundary_accuracy
    }
    
    print(f"  K={k:2d}: Class 0={class_0_count:5d}, Class 1={class_1_count:5d}, Accuracy={boundary_accuracy:.4f}")

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, k in enumerate(k_values):
    ax = axes[idx]
    
    y_test_pred = models_data[k]['predictions']
    class_0_test = X_test[y_test_pred == 0]
    class_1_test = X_test[y_test_pred == 1]
    
    ax.scatter(class_0_test[:, 0], class_0_test[:, 1], 
               c='blue', s=15, alpha=0.5, label='Class 0')
    ax.scatter(class_1_test[:, 0], class_1_test[:, 1], 
               c='red', s=15, alpha=0.5, label='Class 1')
    
    ax.scatter(class_0_train[:, 0], class_0_train[:, 1], 
               c='darkblue', s=120, marker='o', edgecolors='black', 
               linewidth=1.5, zorder=5)
    ax.scatter(class_1_train[:, 0], class_1_train[:, 1], 
               c='darkred', s=120, marker='o', edgecolors='black', 
               linewidth=1.5, zorder=5)
    
    x_boundary = np.linspace(0, 10, 100)
    y_boundary = 11 - x_boundary
    ax.plot(x_boundary, y_boundary, 'g--', linewidth=2.5, label='True Boundary', zorder=4)
    
    ax.set_xlabel('Feature X', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature Y', fontsize=11, fontweight='bold')
    ax.set_title(f'K = {k}\nAccuracy: {models_data[k]["accuracy"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=9, loc='upper left')

plt.suptitle('KNN Decision Boundaries for Varying K Values', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

h = 0.05
xx, yy = np.meshgrid(np.arange(0, 10 + h, h),
                     np.arange(0, 10 + h, h))

for idx, k in enumerate(k_values):
    ax = axes[idx]
    
    knn = models_data[k]['model']
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], 
                colors=['lightblue', 'lightcoral'], alpha=0.6)
    ax.contour(xx, yy, Z, colors='green', linewidths=2.5, levels=[0.5])
    
    y_test_pred = models_data[k]['predictions']
    class_0_test = X_test[y_test_pred == 0]
    class_1_test = X_test[y_test_pred == 1]
    
    ax.scatter(class_0_test[:, 0], class_0_test[:, 1], 
               c='blue', s=10, alpha=0.4)
    ax.scatter(class_1_test[:, 0], class_1_test[:, 1], 
               c='red', s=10, alpha=0.4)
    
    ax.scatter(class_0_train[:, 0], class_0_train[:, 1], 
               c='darkblue', s=150, marker='o', edgecolors='black', 
               linewidth=2, zorder=5, label='Train Class 0')
    ax.scatter(class_1_train[:, 0], class_1_train[:, 1], 
               c='darkred', s=150, marker='o', edgecolors='black', 
               linewidth=2, zorder=5, label='Train Class 1')
    
    x_boundary = np.linspace(0, 10, 100)
    y_boundary = 11 - x_boundary
    ax.plot(x_boundary, y_boundary, 'g--', linewidth=3, 
            label='True Boundary', zorder=4)
    
    ax.set_xlabel('Feature X', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature Y', fontsize=11, fontweight='bold')
    ax.set_title(f'K = {k} (High Resolution)\nAccuracy: {models_data[k]["accuracy"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    
    if idx == 0:
        ax.legend(fontsize=10, loc='upper left')

plt.suptitle('KNN Decision Boundaries (High-Resolution Heatmap View)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("K VALUE IMPACT ANALYSIS")
print("="*80)

accuracies = [models_data[k]['accuracy'] for k in k_values]
class_0_counts = [models_data[k]['class_0_count'] for k in k_values]
class_1_counts = [models_data[k]['class_1_count'] for k in k_values]

print("\nBoundary Classification Accuracy by K:")
print("-" * 50)
for k, acc in zip(k_values, accuracies):
    print(f"  K={k:2d}: {acc:.4f} ({acc*100:.2f}%)")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(k_values, accuracies, 'o-', linewidth=2.5, markersize=10, 
             color='darkblue', label='Boundary Accuracy')
axes[0].set_xlabel('K Value', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Boundary Accuracy vs K', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0.5, 1.0])
for k, acc in zip(k_values, accuracies):
    axes[0].text(k, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=10)

axes[1].plot(k_values, class_0_counts, 'o-', linewidth=2.5, markersize=10, 
             label='Class 0', color='blue')
axes[1].plot(k_values, class_1_counts, 's-', linewidth=2.5, markersize=10, 
             label='Class 1', color='red')
axes[1].set_xlabel('K Value', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Points', fontsize=12, fontweight='bold')
axes[1].set_title('Class Distribution vs K', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

ratios = [c0 / (c0 + c1) if (c0 + c1) > 0 else 0.5 
          for c0, c1 in zip(class_0_counts, class_1_counts)]
axes[2].plot(k_values, ratios, 'd-', linewidth=2.5, markersize=10, 
             color='purple')
axes[2].axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='50-50 Split')
axes[2].set_xlabel('K Value', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Class 0 Ratio', fontsize=12, fontweight='bold')
axes[2].set_title('Class 0 Ratio vs K', fontsize=13, fontweight='bold')
axes[2].set_ylim([0.3, 0.7])
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =====================================================
# A6: KNN CLASSIFICATION FOR SPEECH FLUENCY PROJECT
# Using MFCC_0_mean and MFCC_1_mean Features
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


print("="*80)
print("A6: SPEECH FLUENCY CLASSIFICATION - KNN ANALYSIS")
print("Project: Speech Fluency Classification Using Acoustic Features")
print("="*80)


print("\nStep 1: Loading and Preprocessing Data")

df = pd.read_csv(r'D:\Engineering\4th Semester\Machine Learning\data.csv')

df = df.drop(columns=["filename"])

df = df.apply(pd.to_numeric, errors="coerce")

df = df.fillna(df.mean())
df = df.fillna(0)

features = ["MFCC_0_mean", "MFCC_1_mean"]
X_raw = df[features].values
y = df["Fluency"].values

print(f"  ✓ Dataset loaded: {df.shape[0]} samples")
print(f"  ✓ Features selected: {features}")
print(f"  ✓ Target: Fluency (0=Non-fluent, 1=Fluent)")

print("\nStep 2: Feature Scaling")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"\n  Feature Statistics (Original):")
print(f"    MFCC_0_mean - Mean: {X_raw[:, 0].mean():.2f}, Std: {X_raw[:, 0].std():.2f}")
print(f"    MFCC_1_mean - Mean: {X_raw[:, 1].mean():.2f}, Std: {X_raw[:, 1].std():.2f}")

print(f"\n  Feature Statistics (Scaled):")
print(f"    MFCC_0_mean - Mean: {X_scaled[:, 0].mean():.6f}, Std: {X_scaled[:, 0].std():.2f}")
print(f"    MFCC_1_mean - Mean: {X_scaled[:, 1].mean():.6f}, Std: {X_scaled[:, 1].std():.2f}")

print("\nStep 3: Class Distribution")

class_0_count = np.sum(y == 0)
class_1_count = np.sum(y == 1)

print(f"  Class 0 (Non-fluent): {class_0_count} samples ({(class_0_count/len(y))*100:.1f}%)")
print(f"  Class 1 (Fluent):     {class_1_count} samples ({(class_1_count/len(y))*100:.1f}%)")

print("\n" + "="*80)
print("A3 REPLICA: TRAINING DATA VISUALIZATION")
print("="*80)

class_0_data = X_scaled[y == 0]
class_1_data = X_scaled[y == 1]

plt.figure(figsize=(12, 9))

plt.scatter(class_0_data[:, 0], class_0_data[:, 1], 
            c='blue', s=100, alpha=0.6, marker='o', 
            edgecolors='black', linewidth=1, label='Non-fluent (Class 0)')

plt.scatter(class_1_data[:, 0], class_1_data[:, 1], 
            c='red', s=100, alpha=0.6, marker='o', 
            edgecolors='black', linewidth=1, label='Fluent (Class 1)')

plt.xlabel('MFCC_0_mean (Scaled)', fontsize=13, fontweight='bold')
plt.ylabel('MFCC_1_mean (Scaled)', fontsize=13, fontweight='bold')
plt.title('A3: Speech Fluency Training Data\n({} Total Samples)'.format(len(y)), 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("A4 REPLICA: DENSE TEST SET CLASSIFICATION (K=3)")
print("="*80)

x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5

print(f"\n  Feature Range:")
print(f"    MFCC_0_mean: [{x_min:.2f}, {x_max:.2f}]")
print(f"    MFCC_1_mean: [{y_min:.2f}, {y_max:.2f}]")

h = 0.05  
xx, yy = np.meshgrid(np.arange(x_min, x_max + h, h),
                     np.arange(y_min, y_max + h, h))

X_test_grid = np.c_[xx.ravel(), yy.ravel()]

print(f"  Test grid created: {len(X_test_grid)} points")

k_a4 = 3
knn_a4 = KNeighborsClassifier(n_neighbors=k_a4)
knn_a4.fit(X_scaled, y)

Z_a4 = knn_a4.predict(X_test_grid)
Z_a4_reshaped = Z_a4.reshape(xx.shape)

class_0_pred = X_test_grid[Z_a4 == 0]
class_1_pred = X_test_grid[Z_a4 == 1]

print(f"\n  Predictions (K={k_a4}):")
print(f"    Class 0 regions: {len(class_0_pred)} points")
print(f"    Class 1 regions: {len(class_1_pred)} points")

plt.figure(figsize=(14, 11))

plt.contourf(xx, yy, Z_a4_reshaped, levels=[0, 0.5, 1], 
             colors=['lightblue', 'lightcoral'], alpha=0.5)
plt.contour(xx, yy, Z_a4_reshaped, colors='green', linewidths=2.5, levels=[0.5])

plt.scatter(class_0_pred[:, 0], class_0_pred[:, 1], 
            c='blue', s=8, alpha=0.3)
plt.scatter(class_1_pred[:, 0], class_1_pred[:, 1], 
            c='red', s=8, alpha=0.3)

plt.scatter(class_0_data[:, 0], class_0_data[:, 1], 
            c='darkblue', s=150, marker='o', edgecolors='black', 
            linewidth=1.5, label='Training Non-fluent', zorder=5)

plt.scatter(class_1_data[:, 0], class_1_data[:, 1], 
            c='darkred', s=150, marker='o', edgecolors='black', 
            linewidth=1.5, label='Training Fluent', zorder=5)

plt.xlabel('MFCC_0_mean (Scaled)', fontsize=13, fontweight='bold')
plt.ylabel('MFCC_1_mean (Scaled)', fontsize=13, fontweight='bold')
plt.title(f'A4: KNN Decision Boundary (K={k_a4}) - Dense Test Set\nSpeech Fluency Classification', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("A5 REPLICA: DECISION BOUNDARIES FOR VARYING K VALUES")
print("="*80)

k_values = [1, 3, 5, 7, 10, 15]

models_data = {}
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_scaled, y)
    
    # Predict on training data for accuracy
    y_pred = knn.predict(X_scaled)
    accuracy = np.mean(y_pred == y)
    
    # Predict on test grid
    Z = knn.predict(X_test_grid)
    Z_reshaped = Z.reshape(xx.shape)
    
    class_0_count = np.sum(Z == 0)
    class_1_count = np.sum(Z == 1)
    
    models_data[k] = {
        'model': knn,
        'predictions': Z,
        'Z_reshaped': Z_reshaped,
        'accuracy': accuracy,
        'class_0_count': class_0_count,
        'class_1_count': class_1_count
    }
    
    accuracies.append(accuracy)
    
    print(f"  K={k:2d}: Training Accuracy={accuracy:.4f}, Class 0={class_0_count}, Class 1={class_1_count}")

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.flatten()

for idx, k in enumerate(k_values):
    ax = axes[idx]
    
    Z_reshaped = models_data[k]['Z_reshaped']
    
    ax.contourf(xx, yy, Z_reshaped, levels=[0, 0.5, 1], 
                colors=['lightblue', 'lightcoral'], alpha=0.6)
    ax.contour(xx, yy, Z_reshaped, colors='green', linewidths=2.5, levels=[0.5])
    
    ax.scatter(class_0_data[:, 0], class_0_data[:, 1], 
               c='darkblue', s=120, marker='o', edgecolors='black', 
               linewidth=1.5, zorder=5)
    ax.scatter(class_1_data[:, 0], class_1_data[:, 1], 
               c='darkred', s=120, marker='o', edgecolors='black', 
               linewidth=1.5, zorder=5)
    
    ax.set_xlabel('MFCC_0_mean (Scaled)', fontsize=11, fontweight='bold')
    ax.set_ylabel('MFCC_1_mean (Scaled)', fontsize=11, fontweight='bold')
    ax.set_title(f'K = {k}\nAccuracy: {models_data[k]["accuracy"]:.4f}', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('A5: KNN Decision Boundaries for Varying K Values\nSpeech Fluency Classification', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("COMPARATIVE ANALYSIS - K VALUE IMPACT")
print("="*80)

class_0_counts = [models_data[k]['class_0_count'] for k in k_values]
class_1_counts = [models_data[k]['class_1_count'] for k in k_values]

print("\nAccuracy vs K:")
print("-" * 50)
for k, acc in zip(k_values, accuracies):
    print(f"  K={k:2d}: {acc:.4f} ({acc*100:.2f}%)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(k_values, accuracies, 'o-', linewidth=2.5, markersize=10, 
             color='darkblue')
axes[0].set_xlabel('K Value', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Training Accuracy vs K', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([min(accuracies)-0.05, 1.0])
for k, acc in zip(k_values, accuracies):
    axes[0].text(k, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

axes[1].plot(k_values, class_0_counts, 'o-', linewidth=2.5, markersize=10, 
             label='Non-fluent (Class 0)', color='blue')
axes[1].plot(k_values, class_1_counts, 's-', linewidth=2.5, markersize=10, 
             label='Fluent (Class 1)', color='red')
axes[1].set_xlabel('K Value', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Number of Points', fontsize=12, fontweight='bold')
axes[1].set_title('Predicted Class Distribution vs K', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

ratios = [c0 / (c0 + c1) if (c0 + c1) > 0 else 0.5 
          for c0, c1 in zip(class_0_counts, class_1_counts)]
axes[2].plot(k_values, ratios, 'd-', linewidth=2.5, markersize=10, color='purple')
axes[2].set_xlabel('K Value', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Class 0 Ratio', fontsize=12, fontweight='bold')
axes[2].set_title('Class 0 Ratio vs K', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("FEATURE ANALYSIS")
print("="*80)

correlation = np.corrcoef(X_raw[:, 0], X_raw[:, 1])[0, 1]

print(f"\nFeature Correlation (Original): {correlation:.4f}")

print(f"\nClass-wise Feature Statistics:")
print(f"\n  Non-fluent (Class 0):")
print(f"    MFCC_0_mean - Mean: {X_raw[y==0, 0].mean():.2f}, Std: {X_raw[y==0, 0].std():.2f}")
print(f"    MFCC_1_mean - Mean: {X_raw[y==0, 1].mean():.2f}, Std: {X_raw[y==0, 1].std():.2f}")

print(f"\n  Fluent (Class 1):")
print(f"    MFCC_0_mean - Mean: {X_raw[y==1, 0].mean():.2f}, Std: {X_raw[y==1, 0].std():.2f}")
print(f"    MFCC_1_mean - Mean: {X_raw[y==1, 1].mean():.2f}, Std: {X_raw[y==1, 1].std():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

data_to_plot = [X_raw[y==0, 0], X_raw[y==1, 0], X_raw[y==0, 1], X_raw[y==1, 1]]
bp = axes[0].boxplot(data_to_plot, labels=['Non-fluent\nMFCC_0', 'Fluent\nMFCC_0', 
                                             'Non-fluent\nMFCC_1', 'Fluent\nMFCC_1'])
axes[0].set_ylabel('Value', fontsize=12, fontweight='bold')
axes[0].set_title('Feature Distribution by Class', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].scatter(X_raw[y==0, 0], X_raw[y==0, 1], c='blue', s=50, alpha=0.6, 
                label='Non-fluent', edgecolors='black', linewidth=0.5)
axes[1].scatter(X_raw[y==1, 0], X_raw[y==1, 1], c='red', s=50, alpha=0.6, 
                label='Fluent', edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('MFCC_0_mean', fontsize=12, fontweight='bold')
axes[1].set_ylabel('MFCC_1_mean', fontsize=12, fontweight='bold')
axes[1].set_title('Original Feature Space (Unscaled)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



# =====================================================
# A7: HYPERPARAMETER TUNING FOR KNN CLASSIFIER
# Finding Ideal 'K' Value Using GridSearchCV and RandomizedSearchCV
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


print("="*80)
print("A7: HYPERPARAMETER TUNING FOR KNN CLASSIFIER")
print("Speech Fluency Classification Project")
print("="*80)

print("\nStep 1: Loading and Preprocessing Data")

df = pd.read_csv(r'D:\Engineering\4th Semester\Machine Learning\data.csv')

df = df.drop(columns=["filename"])

df = df.apply(pd.to_numeric, errors="coerce")

df = df.fillna(df.mean())
df = df.fillna(0)

features = ["MFCC_0_mean", "MFCC_1_mean"]
X_raw = df[features].values
y = df["Fluency"].values

print(f"  ✓ Dataset loaded: {df.shape[0]} samples")
print(f"  ✓ Features: {features}")
print(f"  ✓ Target: Fluency (0=Non-fluent, 1=Fluent)")

print("\nStep 2: Feature Scaling")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"  ✓ Features scaled using StandardScaler")
print(f"  ✓ Mean: {X_scaled.mean():.6f}, Std: {X_scaled.std():.2f}")

print("\nStep 3: Train-Test Split")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ✓ Training set: {len(X_train)} samples")
print(f"  ✓ Test set: {len(X_test)} samples")
print(f"  ✓ Train class distribution: Class 0={np.sum(y_train==0)}, Class 1={np.sum(y_train==1)}")
print(f"  ✓ Test class distribution: Class 0={np.sum(y_test==0)}, Class 1={np.sum(y_test==1)}")

print("\n" + "="*80)
print("METHOD 1: GRIDSEARCHCV")
print("="*80)

print("\nPerforming GridSearchCV with K values: 1-30...")

param_grid = {'n_neighbors': range(1, 31)}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(
    knn, 
    param_grid, 
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_k_grid = grid_search.best_params_['n_neighbors']
best_score_grid = grid_search.best_score_

print(f"\n✓ GridSearchCV Complete!")
print(f"  Best K: {best_k_grid}")
print(f"  Best Cross-Validation Score: {best_score_grid:.4f}")

grid_results = pd.DataFrame(grid_search.cv_results_)
k_values_grid = grid_results['param_n_neighbors'].values
cv_scores_grid = grid_results['mean_test_score'].values

print(f"\nTop 5 K values:")
print("-" * 50)
top_5_indices = np.argsort(cv_scores_grid)[-5:][::-1]
for idx in top_5_indices:
    k = k_values_grid[idx]
    score = cv_scores_grid[idx]
    print(f"  K={k:2d}: CV Accuracy = {score:.4f}")

print("\n" + "="*80)
print("METHOD 2: RANDOMIZEDSEARCHCV")
print("="*80)

print("\nPerforming RandomizedSearchCV with K values: 1-30...")

param_dist = {'n_neighbors': np.arange(1, 31)}

random_search = RandomizedSearchCV(
    knn, 
    param_dist, 
    n_iter=20,  # Try 20 random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

best_k_random = random_search.best_params_['n_neighbors']
best_score_random = random_search.best_score_

print(f"\n✓ RandomizedSearchCV Complete!")
print(f"  Best K: {best_k_random}")
print(f"  Best Cross-Validation Score: {best_score_random:.4f}")

random_results = pd.DataFrame(random_search.cv_results_)
k_values_random = random_results['param_n_neighbors'].values
cv_scores_random = random_results['mean_test_score'].values

print(f"\nTop 5 K values from Random Search:")
print("-" * 50)
top_5_random_indices = np.argsort(cv_scores_random)[-5:][::-1]
for idx in top_5_random_indices:
    k = k_values_random[idx]
    score = cv_scores_random[idx]
    print(f"  K={k:2d}: CV Accuracy = {score:.4f}")

print("\n" + "="*80)
print("COMPARISON: GridSearchCV vs RandomizedSearchCV")
print("="*80)

print(f"\nGridSearchCV:")
print(f"  ✓ Best K: {best_k_grid}")
print(f"  ✓ Best CV Score: {best_score_grid:.4f}")
print(f"  ✓ Search Strategy: Exhaustive (tests all values)")
print(f"  ✓ Time Complexity: Higher (tested all 30 values)")

print(f"\nRandomizedSearchCV:")
print(f"  ✓ Best K: {best_k_random}")
print(f"  ✓ Best CV Score: {best_score_random:.4f}")
print(f"  ✓ Search Strategy: Random sampling (20 iterations)")
print(f"  ✓ Time Complexity: Lower (faster)")

print("\nStep 5: Visualizing Search Results...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].plot(k_values_grid, cv_scores_grid, 'o-', linewidth=2.5, markersize=8, color='darkblue')
axes[0, 0].axvline(x=best_k_grid, color='red', linestyle='--', linewidth=2, label=f'Best K={best_k_grid}')
axes[0, 0].set_xlabel('K Value', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_title('GridSearchCV: CV Accuracy vs K', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=11)
axes[0, 0].set_ylim([min(cv_scores_grid)-0.05, max(cv_scores_grid)+0.05])

axes[0, 1].plot(k_values_random, cv_scores_random, 's-', linewidth=2.5, markersize=8, color='darkgreen')
axes[0, 1].axvline(x=best_k_random, color='red', linestyle='--', linewidth=2, label=f'Best K={best_k_random}')
axes[0, 1].set_xlabel('K Value', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_title('RandomizedSearchCV: CV Accuracy vs K', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=11)
axes[0, 1].set_ylim([min(cv_scores_random)-0.05, max(cv_scores_random)+0.05])

axes[1, 0].plot(k_values_grid, cv_scores_grid, 'o-', linewidth=2.5, markersize=8, 
               label='GridSearchCV', color='darkblue')
axes[1, 0].plot(k_values_random, cv_scores_random, 's-', linewidth=2.5, markersize=8, 
               label='RandomizedSearchCV', color='darkgreen', alpha=0.7)
axes[1, 0].axvline(x=best_k_grid, color='blue', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 0].axvline(x=best_k_random, color='green', linestyle='--', linewidth=2, alpha=0.7)
axes[1, 0].set_xlabel('K Value', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Comparison: GridSearchCV vs RandomizedSearchCV', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=11)

top_10_indices = np.argsort(cv_scores_grid)[-10:][::-1]
axes[1, 1].bar(range(10), cv_scores_grid[top_10_indices], color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 1].set_xticks(range(10))
axes[1, 1].set_xticklabels(k_values_grid[top_10_indices].astype(int), fontsize=10)
axes[1, 1].set_xlabel('K Value (Top 10)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('CV Accuracy', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Top 10 K Values from GridSearchCV', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, (k, score) in enumerate(zip(k_values_grid[top_10_indices].astype(int), cv_scores_grid[top_10_indices])):
    axes[1, 1].text(i, score + 0.005, f'{score:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("STEP 6: TRAINING FINAL MODEL")
print("="*80)

final_knn = KNeighborsClassifier(n_neighbors=best_k_grid)
final_knn.fit(X_train, y_train)

y_train_pred = final_knn.predict(X_train)
y_test_pred = final_knn.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_prec = precision_score(y_train, y_train_pred, average='weighted')
test_prec = precision_score(y_test, y_test_pred, average='weighted')

train_rec = recall_score(y_train, y_train_pred, average='weighted')
test_rec = recall_score(y_test, y_test_pred, average='weighted')

train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"\nFinal Model Performance (K={best_k_grid}):\n")

print("Training Metrics:")
print(f"  Accuracy:  {train_acc:.4f}")
print(f"  Precision: {train_prec:.4f}")
print(f"  Recall:    {train_rec:.4f}")
print(f"  F1 Score:  {train_f1:.4f}")

print(f"\nTest Metrics:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_prec:.4f}")
print(f"  Recall:    {test_rec:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print(f"\nTrain Confusion Matrix:")
print(cm_train)

print(f"\nTest Confusion Matrix:")
print(cm_test)

print(f"\n" + "="*80)
print("TEST SET CLASSIFICATION REPORT (K={})".format(best_k_grid))
print("="*80)
print(classification_report(y_test, y_test_pred, 
                          target_names=['Non-fluent', 'Fluent'], digits=4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-fluent', 'Fluent'],
            yticklabels=['Non-fluent', 'Fluent'],
            cbar_kws={'label': 'Count'})
axes[0].set_title(f'Train Confusion Matrix (K={best_k_grid})', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_xlabel('Predicted', fontsize=12)

sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Non-fluent', 'Fluent'],
            yticklabels=['Non-fluent', 'Fluent'],
            cbar_kws={'label': 'Count'})
axes[1].set_title(f'Test Confusion Matrix (K={best_k_grid})', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_xlabel('Predicted', fontsize=12)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("STEP 9: LEARNING CURVE ANALYSIS")
print("="*80)

train_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
val_scores = []

for train_size in train_sizes:
    n_samples = int(train_size * len(X_train))
    X_partial = X_train[:n_samples]
    y_partial = y_train[:n_samples]
    
    knn_temp = KNeighborsClassifier(n_neighbors=best_k_grid)
    knn_temp.fit(X_partial, y_partial)
    
    train_scores.append(knn_temp.score(X_partial, y_partial))
    val_scores.append(knn_temp.score(X_test, y_test))

print("\nLearning Curve (Train Size % vs Accuracy):")
print("-" * 50)
for ts, tr_score, val_score in zip(train_sizes*100, train_scores, val_scores):
    print(f"  Train Size: {ts:5.1f}% - Train Acc: {tr_score:.4f}, Val Acc: {val_score:.4f}")

plt.figure(figsize=(12, 7))

plt.plot(train_sizes*100, train_scores, 'o-', linewidth=2.5, markersize=8, 
         label='Training Accuracy', color='darkblue')
plt.plot(train_sizes*100, val_scores, 's-', linewidth=2.5, markersize=8, 
         label='Validation Accuracy', color='darkred')

plt.xlabel('Training Set Size (%)', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title(f'Learning Curve for KNN (K={best_k_grid})', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim([0.4, 1.0])

plt.tight_layout()
plt.show()

