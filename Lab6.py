import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("data.csv")

df = df.drop(columns=["filename"])

target = "Fluency"

#A4
def equal_width_binning(data, bins=4):
    min_val = min(data)
    max_val = max(data)
    width = (max_val - min_val) / bins
    binned = []
    for value in data:
        bin_index = int((value - min_val) / width)
        if bin_index == bins:
            bin_index -= 1
        binned.append(bin_index)
    return binned


#A1
def entropy(target_col):
    total = len(target_col)
    counts = Counter(target_col)
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * np.log2(p)
    return ent


#A2
def gini_index(target_col):
    total = len(target_col)
    counts = Counter(target_col)
    gini = 1
    for count in counts.values():
        p = count / total
        gini -= p**2
    return gini


#A3
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = 0
    for v in values:
        subset = data[data[feature] == v]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset[target])
    return total_entropy - weighted_entropy


#A3
def find_root_node(data, target):
    features = data.columns.drop(target)
    gains = {}
    for feature in features:
        gains[feature] = information_gain(data, feature, target)
    root = max(gains, key=gains.get)
    return root, gains


#A5
def build_tree(data, target, features):

    if len(set(data[target])) == 1:
        return data[target].iloc[0]

    if len(features) == 0:
        return data[target].mode()[0]

    root = find_root_node(data, target)[0]

    tree = {root: {}}

    for value in data[root].unique():

        subset = data[data[root] == value]

        if subset.empty:
            tree[root][value] = data[target].mode()[0]

        else:
            remaining = features.drop(root)
            subtree = build_tree(subset, target, remaining)
            tree[root][value] = subtree

    return tree


print("Entropy:", entropy(df[target]))
print("Gini Index:", gini_index(df[target]))

root, gains = find_root_node(df, target)

print("Root Node:", root)

features = df.columns.drop(target)

tree = build_tree(df, target, features)

print("Decision Tree:", tree)


#A6
X = df.drop(target, axis=1)
y = df[target]

model = DecisionTreeClassifier()

model.fit(X, y)

plt.figure(figsize=(12,8))

plot_tree(model,
          feature_names=X.columns,
          class_names=["0","1"],
          filled=True)

plt.show()


#A7
X2 = df[["MFCC_0_mean","MFCC_1_mean"]]
y2 = df[target]

model2 = DecisionTreeClassifier()

model2.fit(X2,y2)

x_min, x_max = X2.iloc[:,0].min()-1, X2.iloc[:,0].max()+1
y_min, y_max = X2.iloc[:,1].min()-1, X2.iloc[:,1].max()+1

xx, yy = np.meshgrid(
    np.arange(x_min,x_max,0.1),
    np.arange(y_min,y_max,0.1)
)

Z = model2.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx,yy,Z,alpha=0.3)

plt.scatter(X2.iloc[:,0],X2.iloc[:,1],c=y2)

plt.xlabel("MFCC_0_mean")
plt.ylabel("MFCC_1_mean")

plt.show()