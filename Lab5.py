import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np

df = pd.read_csv('data.csv')

df = df.drop(columns=['filename'], errors='ignore')

df = df.replace([np.inf, -np.inf], np.nan)

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    non_zero = y_true != 0
    if np.sum(non_zero) == 0:
        return np.nan
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


# Single feature
X = df[['MFCC_0_mean']]
y = df['Fluency']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Impute (median)
imputer_single = SimpleImputer(strategy='median')
X_train = imputer_single.fit_transform(X_train)
X_test = imputer_single.transform(X_test)

# Train model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

# Metrics - Train
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mape = safe_mape(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Metrics - Test
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mape = safe_mape(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("A2: TRAIN SET METRICS")
print(f"MSE: {train_mse}")
print(f"RMSE: {train_rmse}")
print(f"MAPE: {train_mape}")
print(f"R2: {train_r2}")

print("\nA2: TEST SET METRICS")
print(f"MSE: {test_mse}")
print(f"RMSE: {test_rmse}")
print(f"MAPE: {test_mape}")
print(f"R2: {test_r2}")


# All features except target
X_multi = df.drop(columns=['Fluency'])
y_multi = df['Fluency']

# Drop columns that are completely NaN
X_multi = X_multi.dropna(axis=1, how='all')

# Train-test split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Impute median (fit only on training data)
imputer_multi = SimpleImputer(strategy='median')
X_train_m = imputer_multi.fit_transform(X_train_m)
X_test_m = imputer_multi.transform(X_test_m)

# Train model
reg_multi = LinearRegression()
reg_multi.fit(X_train_m, y_train_m)

# Predictions
y_train_pred_m = reg_multi.predict(X_train_m)
y_test_pred_m = reg_multi.predict(X_test_m)

# Metrics - Train
train_mse_m = mean_squared_error(y_train_m, y_train_pred_m)
train_rmse_m = np.sqrt(train_mse_m)
train_mape_m = safe_mape(y_train_m, y_train_pred_m)
train_r2_m = r2_score(y_train_m, y_train_pred_m)

# Metrics - Test
test_mse_m = mean_squared_error(y_test_m, y_test_pred_m)
test_rmse_m = np.sqrt(test_mse_m)
test_mape_m = safe_mape(y_test_m, y_test_pred_m)
test_r2_m = r2_score(y_test_m, y_test_pred_m)

print("\nA3: MULTIPLE FEATURE TRAIN SET METRICS")
print(f"MSE: {train_mse_m}")
print(f"RMSE: {train_rmse_m}")
print(f"MAPE: {train_mape_m}")
print(f"R2: {train_r2_m}")

print("\nA3: MULTIPLE FEATURE TEST SET METRICS")
print(f"MSE: {test_mse_m}")
print(f"RMSE: {test_rmse_m}")
print(f"MAPE: {test_mape_m}")
print(f"R2: {test_r2_m}")


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Remove target column
X_cluster = df.drop(columns=['Fluency'])

# Replace inf values
X_cluster = X_cluster.replace([np.inf, -np.inf], np.nan)

# Drop columns that are fully NaN
X_cluster = X_cluster.dropna(axis=1, how='all')

# Median imputation
imputer_cluster = SimpleImputer(strategy='median')
X_cluster = imputer_cluster.fit_transform(X_cluster)

# Feature Scaling (VERY IMPORTANT)
scaler = StandardScaler()
X_cluster = scaler.fit_transform(X_cluster)

kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init='auto')
kmeans_2.fit(X_cluster)

labels_2 = kmeans_2.labels_

print("\nA4: KMeans (k=2)")
print("Cluster Centers shape:", kmeans_2.cluster_centers_.shape)


sil_score = silhouette_score(X_cluster, labels_2)
ch_score = calinski_harabasz_score(X_cluster, labels_2)
db_score = davies_bouldin_score(X_cluster, labels_2)

print("\nA5: Clustering Metrics (k=2)")
print(f"Silhouette Score: {sil_score}")
print(f"Calinski-Harabasz Score: {ch_score}")
print(f"Davies-Bouldin Index: {db_score}")


k_values = range(2, 11)
sil_scores = []
ch_scores = []
db_scores = []
distortions = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_cluster)
    labels = kmeans.labels_
    
    sil_scores.append(silhouette_score(X_cluster, labels))
    ch_scores.append(calinski_harabasz_score(X_cluster, labels))
    db_scores.append(davies_bouldin_score(X_cluster, labels))
    distortions.append(kmeans.inertia_)


plt.figure()
plt.plot(k_values, sil_scores)
plt.title("Silhouette Score vs k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()

plt.figure()
plt.plot(k_values, ch_scores)
plt.title("Calinski-Harabasz Score vs k")
plt.xlabel("k")
plt.ylabel("CH Score")
plt.show()

plt.figure()
plt.plot(k_values, db_scores)
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("k")
plt.ylabel("DB Index")
plt.show()

# Elbow Plot

plt.figure()
plt.plot(k_values, distortions)
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Distortion (Inertia)")
plt.show()