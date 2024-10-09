import os
import pandas as pd

DATASET_DIRECTORY = "./datasets/"

symbol_dfs = {}

for filename in os.listdir(DATASET_DIRECTORY):
    if filename.endswith(".csv"):
        file_path = os.path.join(DATASET_DIRECTORY, filename)

        # Read the csv file using pandas
        df = pd.read_csv(file_path)

        symbol = filename.split(".")[0]
        symbol_dfs[symbol] = df

for symbol in list(symbol_dfs.keys()):
    df = symbol_dfs[symbol]

    if "Date" in df.columns:
        df.drop(columns=["Date"], inplace=True)

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])

    # Remove if the starting year is greater than 2008
    if df["Time"][0].year > 2008:
        del symbol_dfs[symbol]
    else:
        symbol_dfs[symbol] = df

symbols = list(symbol_dfs.keys())

min_len = min([len(symbol_dfs[symbol]) for symbol in symbol_dfs])

for symbol, df in symbol_dfs.items():
    symbol_dfs[symbol] = df.iloc[-min_len:]
    symbol_dfs[symbol] = symbol_dfs[symbol].reset_index(drop=True)

import numpy as np
from sklearn.preprocessing import StandardScaler

for symbol, df in symbol_dfs.items():

    # Calculate Log Return from Close price
    df["LogR"] = np.log(df["Close"]) - np.log(df["Close"].shift(1))

    # Median filtering
    df["LogR"] = df["LogR"].fillna(df["LogR"].median())

    # Normalize the LogR column
    scaler = StandardScaler()
    df["LogR"] = scaler.fit_transform(df[["LogR"]])

    symbol_dfs[symbol] = df

import numpy as np

# List to hold LogR columns from all symbols
logr_list = []

# Loop through each symbol and its corresponding DataFrame
for symbol, df in symbol_dfs.items():
    # Extract the LogR column and convert it to a numpy array
    logr_column = df["LogR"].values  # This converts the 'LogR' column to a numpy array
    logr_list.append(logr_column)

# Stack the list of arrays into a 2D numpy array (symbol_n * column_length)
returns_data = np.vstack(logr_list)

from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Elbow Method to find optimal number of clusters
max_clusters = 14
silhouette_scores = []

for n_clusters in range(2, max_clusters + 1):
    print("Cluster number: ", n_clusters)
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
    clusters = model.fit_predict(returns_data)

    # Store inertia (sum of distances to the closest cluster center)
    clusters = model.fit_predict(returns_data)

    print("Model fitting done!")

    # Calculate silhouette score
    score = silhouette_score(
        returns_data, clusters, metric="euclidean"
    )  # Silhouette score with Euclidean distance
    silhouette_scores.append(score)

    print("sulhouette score calculated")

    # Group symbols by cluster
    cluster_dict = {i: [] for i in range(n_clusters)}
    for i, cluster in enumerate(clusters):
        cluster_dict[cluster].append(symbols[i])

    # Print symbols for each cluster
    for cluster, sym_list in cluster_dict.items():
        print(f"Cluster {cluster}: {', '.join(sym_list)}")

optimal_clusters = (
    np.argmax(silhouette_scores) + 2
)  # Adding 2 because the range starts from 2 clusters

print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")

# Refit the model using the optimal number of clusters
model = TimeSeriesKMeans(n_clusters=optimal_clusters, metric="dtw", random_state=0)
optimal_clusters_labels = model.fit_predict(returns_data)

# Group symbols by optimal clusters and print
optimal_cluster_dict = {i: [] for i in range(optimal_clusters)}
for i, cluster in enumerate(optimal_clusters_labels):
    optimal_cluster_dict[cluster].append(symbols[i])

# Print symbols for each optimal cluster
for cluster, sym_list in optimal_cluster_dict.items():
    print(f"Cluster {cluster}: {', '.join(sym_list)}")
