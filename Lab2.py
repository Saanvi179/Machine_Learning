import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank

# -----------------------------
# Load the dataset
# -----------------------------

df = pd.read_excel(
    r"d:\Engineering\4th Semester\Machine Learning\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

# -----------------------------
# A1: Form X and y
# -----------------------------

X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
y = df["Payment (Rs)"].values.reshape(-1, 1)

print("Feature matrix shape:", X.shape)
print("Output vector shape:", y.shape)

# -----------------------------
# A1: Rank of feature matrix
# -----------------------------

rank_X = matrix_rank(X)
print("Rank of feature matrix X:", rank_X)

# -----------------------------
# A1: Using Pseudo-Inverse to find prices
# -----------------------------

X_pinv = np.linalg.pinv(X)
w = X_pinv @ y

print("\nLearned Product Prices:")
print("Candy Price  :", w[0, 0])
print("Mango Price  :", w[1, 0])
print("Milk Price   :", w[2, 0])

# -----------------------------
# A2: Create RICH / POOR labels from true Payment
# -----------------------------

y_true_class = np.where(df["Payment (Rs)"] > 200, 1, 0)
df["True_Class"] = y_true_class

# -----------------------------
# A2: Predict Payment using learned model
# -----------------------------

y_pred_payment = X @ w   # Predicted payment using linear model

# -----------------------------
# A2: Classify using threshold (Manual Classifier)
# -----------------------------

y_pred_class = np.where(y_pred_payment > 200, 1, 0)

df["Predicted_Class"] = y_pred_class

# -----------------------------
# A2: Evaluate Manually (Accuracy)
# -----------------------------

accuracy = np.mean(y_pred_class.flatten() == y_true_class)

print("\nTrue Class vs Predicted Class:")
print(df[["Payment (Rs)", "True_Class", "Predicted_Class"]])

print("\nClassification Accuracy:", accuracy)

# -----------------------------
# A2: Predict for a New Customer
# -----------------------------

new_customer = np.array([[5, 2, 4]])  # 5 candies, 2kg mango, 4 milk

pred_payment = new_customer @ w

if pred_payment > 200:
    print("\nNew customer predicted as: RICH")
else:
    print("\nNew customer predicted as: POOR")


# =============================
# A3: IRCTC Stock Price Analysis
# =============================

import time
import matplotlib.pyplot as plt

# -----------------------------
# Load IRCTC Stock Price sheet
# -----------------------------

df_stock = pd.read_excel(
    r"d:\Engineering\4th Semester\Machine Learning\Lab Session Data.xlsx",
    sheet_name="IRCTC Stock Price"
)

# Assume:
# Column D = Price
# Column I = Chg%
# Date column name assumed as "Date"

price = df_stock.iloc[:, 3].values        # Column D
chg = df_stock.iloc[:, 8].values          # Column I
dates = pd.to_datetime(df_stock.iloc[:, 0])  # First column as Date

# -----------------------------
# 1. Mean and Variance using NumPy
# -----------------------------

np_mean = np.mean(price)
np_var = np.var(price)

print("\nA3: NumPy Mean:", np_mean)
print("A3: NumPy Variance:", np_var)

# -----------------------------
# 2. Own Functions for Mean and Variance
# -----------------------------

def my_mean(x):
    s = 0.0
    n = len(x)
    for i in x:
        s += i
    return s / n

def my_var(x):
    m = my_mean(x)
    s = 0.0
    n = len(x)
    for i in x:
        s += (i - m) ** 2
    return s / n

my_mean_val = my_mean(price)
my_var_val = my_var(price)

print("\nMy Mean:", my_mean_val)
print("My Variance:", my_var_val)

# -----------------------------
# 2b. Time Complexity Comparison
# -----------------------------

runs = 10

# NumPy timing
np_times = []
for _ in range(runs):
    t0 = time.time()
    _ = np.mean(price)
    _ = np.var(price)
    t1 = time.time()
    np_times.append(t1 - t0)

# My function timing
my_times = []
for _ in range(runs):
    t0 = time.time()
    _ = my_mean(price)
    _ = my_var(price)
    t1 = time.time()
    my_times.append(t1 - t0)

print("\nAverage NumPy Time:", np.mean(np_times))
print("Average My Function Time:", np.mean(my_times))

# -----------------------------
# 3. Wednesday Sample Mean
# -----------------------------

df_stock["Day"] = dates.dt.day_name()
df_stock["Month"] = dates.dt.month

wednesday_prices = df_stock[df_stock["Day"] == "Wednesday"].iloc[:, 3].values

wed_mean = np.mean(wednesday_prices)

print("\nPopulation Mean:", np_mean)
print("Wednesday Sample Mean:", wed_mean)

# -----------------------------
# 4. April Sample Mean
# -----------------------------

april_prices = df_stock[df_stock["Month"] == 4].iloc[:, 3].values

april_mean = np.mean(april_prices)

print("\nApril Sample Mean:", april_mean)

# -----------------------------
# 5. Probability of Making a Loss
# -----------------------------

loss_count = np.sum(list(map(lambda x: x < 0, chg)))
total_count = len(chg)

prob_loss = loss_count / total_count

print("\nProbability of Making a Loss:", prob_loss)

# -----------------------------
# 6. Probability of Making Profit on Wednesday
# -----------------------------

profit_and_wed = df_stock[(df_stock["Day"] == "Wednesday") & (chg > 0)]
prob_profit_wed = len(profit_and_wed) / total_count

print("\nProbability of Profit on Wednesday:", prob_profit_wed)

# -----------------------------
# 7. Conditional Probability: Profit | Wednesday
# -----------------------------

total_wed = len(df_stock[df_stock["Day"] == "Wednesday"])
profit_given_wed = len(profit_and_wed) / total_wed

print("\nConditional Probability P(Profit | Wednesday):", profit_given_wed)

# -----------------------------
# 8. Scatter Plot: Chg% vs Day of Week
# -----------------------------

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

day_numeric = df_stock["Day"].map(day_map)

plt.figure(figsize=(8, 5))
plt.scatter(day_numeric, chg)
plt.xlabel("Day of Week (0=Mon, ..., 6=Sun)")
plt.ylabel("Chg%")
plt.title("Scatter Plot of Chg% vs Day of Week")
plt.grid(True)
plt.show()


# =============================
# A4: Data Exploration on thyroid0387_UCI
# =============================

# Load thyroid dataset
df_thyroid = pd.read_excel(
    r"d:\Engineering\4th Semester\Machine Learning\Lab Session Data.xlsx",
    sheet_name="thyroid0387_UCI"
)

print("\nA4: Thyroid Dataset Loaded")
print("Shape of dataset:", df_thyroid.shape)

# -----------------------------
# 1. Study each attribute and datatype
# -----------------------------

print("\nAttribute Datatypes:")
print(df_thyroid.dtypes)

# Separate numeric and categorical columns
numeric_cols = df_thyroid.select_dtypes(include=[np.number]).columns
categorical_cols = df_thyroid.select_dtypes(exclude=[np.number]).columns

print("\nNumeric Attributes:")
print(list(numeric_cols))

print("\nCategorical Attributes:")
print(list(categorical_cols))

# -----------------------------
# 2. Encoding scheme for categorical attributes
# -----------------------------

print("\nEncoding Scheme Suggestion:")

for col in categorical_cols:
    unique_vals = df_thyroid[col].dropna().unique()
    print(f"\nAttribute: {col}")
    print("Unique Values:", unique_vals)

    # Simple rule-based suggestion
    if len(unique_vals) <= 2:
        print("Suggested Encoding: Label Encoding (Binary / Ordinal-like)")
    else:
        print("Suggested Encoding: One-Hot Encoding (Nominal)")

# -----------------------------
# 3. Data range for numeric variables
# -----------------------------

print("\nData Range for Numeric Attributes:")

for col in numeric_cols:
    col_min = df_thyroid[col].min()
    col_max = df_thyroid[col].max()
    print(f"{col}: Min = {col_min}, Max = {col_max}")

# -----------------------------
# 4. Missing values in each attribute
# -----------------------------

print("\nMissing Values in Each Attribute:")

missing_counts = df_thyroid.isna().sum()

for col in df_thyroid.columns:
    print(f"{col}: {missing_counts[col]} missing values")

# -----------------------------
# 5. Outlier Detection using IQR method
# -----------------------------

print("\nOutlier Analysis (IQR Method):")

outlier_summary = {}

for col in numeric_cols:
    Q1 = df_thyroid[col].quantile(0.25)
    Q3 = df_thyroid[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_thyroid[(df_thyroid[col] < lower_bound) | (df_thyroid[col] > upper_bound)]
    outlier_count = outliers.shape[0]

    outlier_summary[col] = outlier_count

    print(f"{col}: Number of outliers = {outlier_count}")

# -----------------------------
# 6. Mean and Variance for Numeric Variables
# -----------------------------

print("\nMean and Variance for Numeric Attributes:")

for col in numeric_cols:
    mean_val = df_thyroid[col].mean()
    var_val = df_thyroid[col].var()
    std_val = df_thyroid[col].std()

    print(f"{col}: Mean = {mean_val}, Variance = {var_val}, Std Dev = {std_val}")


# =============================
# A5: Similarity Measures (Jaccard & SMC)
# =============================

# Use the same thyroid dataset loaded in A4: df_thyroid

print("\nA5: Similarity Measures using first two observation vectors")

# -----------------------------
# Step 1: Identify Binary Attributes
# -----------------------------

binary_cols = []

for col in df_thyroid.columns:
    unique_vals = df_thyroid[col].dropna().unique()

    # Convert possible 't'/'f' to 1/0 for checking
    if set(unique_vals).issubset({0, 1}) and len(unique_vals) <= 2:
        binary_cols.append(col)

print("\nBinary Attributes Considered:")
print(binary_cols)

# -----------------------------
# Step 2: Extract First Two Vectors (Binary Only)
# -----------------------------

v1 = df_thyroid.loc[0, binary_cols].values
v2 = df_thyroid.loc[1, binary_cols].values

print("\nFirst Binary Vector:")
print(v1)

print("\nSecond Binary Vector:")
print(v2)

# -----------------------------
# Step 3: Compute f00, f01, f10, f11
# -----------------------------

f00 = 0  # both 0
f01 = 0  # first 0, second 1
f10 = 0  # first 1, second 0
f11 = 0  # both 1

for a, b in zip(v1, v2):
    if a == 0 and b == 0:
        f00 += 1
    elif a == 0 and b == 1:
        f01 += 1
    elif a == 1 and b == 0:
        f10 += 1
    elif a == 1 and b == 1:
        f11 += 1

print("\nf00 (0,0) count:", f00)
print("f01 (0,1) count:", f01)
print("f10 (1,0) count:", f10)
print("f11 (1,1) count:", f11)

# -----------------------------
# Step 4: Compute Jaccard Coefficient (JC)
# -----------------------------

JC = f11 / (f01 + f10 + f11)

# -----------------------------
# Step 5: Compute Simple Matching Coefficient (SMC)
# -----------------------------

SMC = (f11 + f00) / (f00 + f01 + f10 + f11)

print("\nJaccard Coefficient (JC):", JC)
print("Simple Matching Coefficient (SMC):", SMC)

# -----------------------------
# Step 6: Interpretation
# -----------------------------

print("\nInterpretation:")
print("Jaccard ignores f00 (both zero matches) and focuses on presence (1s).")
print("SMC counts both 1-1 and 0-0 matches as similarity.")

if JC < SMC:
    print("SMC is larger because it includes f00, while Jaccard ignores f00.")
else:
    print("Jaccard is larger, indicating strong similarity in presence attributes.")


# =============================
# A6: Cosine Similarity Measure
# =============================

print("\nA6: Cosine Similarity between first two complete observation vectors")

# Use the same thyroid dataset: df_thyroid

# -----------------------------
# Step 1: Take first two observations
# -----------------------------

row1 = df_thyroid.iloc[0].copy()
row2 = df_thyroid.iloc[1].copy()

# -----------------------------
# Step 2: Convert categorical to numeric
# -----------------------------

def convert_to_numeric(x):
    if isinstance(x, str):
        if x.lower() == 't':
            return 1.0
        elif x.lower() == 'f':
            return 0.0
        else:
            return 0.0   # for other categorical / missing
    elif pd.isna(x):
        return 0.0
    else:
        return float(x)

vec1 = row1.apply(convert_to_numeric).values.astype(float)
vec2 = row2.apply(convert_to_numeric).values.astype(float)

print("\nVector A (first observation):")
print(vec1)

print("\nVector B (second observation):")
print(vec2)

# -----------------------------
# Step 3: Compute Dot Product
# -----------------------------

dot_product = np.dot(vec1, vec2)

# -----------------------------
# Step 4: Compute Norms
# -----------------------------

norm_A = np.sqrt(np.sum(vec1 ** 2))
norm_B = np.sqrt(np.sum(vec2 ** 2))

# -----------------------------
# Step 5: Compute Cosine Similarity
# -----------------------------

cosine_similarity = dot_product / (norm_A * norm_B)

print("\nDot Product <A, B>:", dot_product)
print("||A|| (Norm of A):", norm_A)
print("||B|| (Norm of B):", norm_B)

print("\nCosine Similarity between A and B:", cosine_similarity)


# =============================
# A7: Heatmap of Similarities (JC, SMC, COS)
# =============================

import seaborn as sns
import matplotlib.pyplot as plt

print("\nA7: Heatmap of Similarities for first 20 observation vectors")

# Take first 20 observations
df_20 = df_thyroid.iloc[:20].copy()

# ---------- Helper: Prepare Binary Matrix ----------

binary_cols = []
for col in df_20.columns:
    unique_vals = df_20[col].dropna().unique()
    if set(unique_vals).issubset({0, 1}) and len(unique_vals) <= 2:
        binary_cols.append(col)

binary_data = df_20[binary_cols].values

# ---------- Helper: Prepare Numeric Matrix for Cosine ----------

def convert_row_to_numeric(row):
    def convert(x):
        if isinstance(x, str):
            if x.lower() == 't':
                return 1.0
            elif x.lower() == 'f':
                return 0.0
            else:
                return 0.0
        elif pd.isna(x):
            return 0.0
        else:
            return float(x)
    return row.apply(convert).values.astype(float)

numeric_data = np.array([convert_row_to_numeric(df_20.iloc[i]) for i in range(20)])

# ---------- Functions for JC, SMC, COS ----------

def jaccard(v1, v2):
    f11 = f10 = f01 = 0
    for a, b in zip(v1, v2):
        if a == 1 and b == 1:
            f11 += 1
        elif a == 1 and b == 0:
            f10 += 1
        elif a == 0 and b == 1:
            f01 += 1
    if (f11 + f10 + f01) == 0:
        return 0
    return f11 / (f11 + f10 + f01)

def smc(v1, v2):
    f11 = f10 = f01 = f00 = 0
    for a, b in zip(v1, v2):
        if a == 1 and b == 1:
            f11 += 1
        elif a == 1 and b == 0:
            f10 += 1
        elif a == 0 and b == 1:
            f01 += 1
        elif a == 0 and b == 0:
            f00 += 1
    return (f11 + f00) / (f11 + f10 + f01 + f00)

def cosine_sim(a, b):
    dot = np.dot(a, b)
    norm_a = np.sqrt(np.sum(a ** 2))
    norm_b = np.sqrt(np.sum(b ** 2))
    return dot / (norm_a * norm_b)

# ---------- Build Similarity Matrices ----------

JC_mat = np.zeros((20, 20))
SMC_mat = np.zeros((20, 20))
COS_mat = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        JC_mat[i, j] = jaccard(binary_data[i], binary_data[j])
        SMC_mat[i, j] = smc(binary_data[i], binary_data[j])
        COS_mat[i, j] = cosine_sim(numeric_data[i], numeric_data[j])

# ---------- Plot Heatmaps ----------

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.heatmap(JC_mat, cmap="viridis")
plt.title("Jaccard Coefficient")

plt.subplot(1, 3, 2)
sns.heatmap(SMC_mat, cmap="viridis")
plt.title("Simple Matching Coefficient")

plt.subplot(1, 3, 3)
sns.heatmap(COS_mat, cmap="viridis")
plt.title("Cosine Similarity")

plt.tight_layout()
plt.show()

# =============================
# A8: Data Imputation
# =============================

print("\nA8: Data Imputation")

df_imputed = df_thyroid.copy()

numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
categorical_cols = df_imputed.select_dtypes(exclude=[np.number]).columns

# Impute numeric columns
for col in numeric_cols:
    col_data = df_imputed[col]
    
    # Detect outliers using IQR
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    has_outliers = ((col_data < lower) | (col_data > upper)).any()
    
    if has_outliers:
        fill_value = col_data.median()
        method = "Median"
    else:
        fill_value = col_data.mean()
        method = "Mean"
    
    df_imputed[col] = col_data.fillna(fill_value)
    print(f"{col}: Missing filled using {method}")

# Impute categorical columns using mode
for col in categorical_cols:
    mode_val = df_imputed[col].mode()[0]
    df_imputed[col] = df_imputed[col].fillna(mode_val)
    print(f"{col}: Missing filled using Mode")

# =============================
# A9: Data Normalization / Scaling
# =============================

print("\nA9: Data Normalization")

df_normalized_minmax = df_imputed.copy()
df_normalized_zscore = df_imputed.copy()

for col in numeric_cols:
    x = df_imputed[col].values
    
    # Min-Max Normalization
    xmin = np.min(x)
    xmax = np.max(x)
    df_normalized_minmax[col] = (x - xmin) / (xmax - xmin)
    
    # Z-score Normalization
    mu = np.mean(x)
    sigma = np.std(x)
    df_normalized_zscore[col] = (x - mu) / sigma

print("\nMin-Max Normalization applied on numeric attributes.")
print("Z-score Normalization applied on numeric attributes.")

