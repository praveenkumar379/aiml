import numpy as np
import matplotlib.pyplot as plt

# Sample data: CustomerID, Number of Purchases, and Total Reward Points
# Generate 1000 random customer IDs
np.random.seed(42)  # For reproducibility
customer_ids = np.random.randint(1000, 9999, size=1000)

# Generate random data for Number of Purchases and Total Reward Points
number_of_purchases = np.random.randint(1, 1001, size=1000)
product_prices = np.random.uniform(300, 20000, size=1000)
total_reward_points = (number_of_purchases * product_prices * 0.01).astype(int)

# Combine the data into a single array
X = np.column_stack((customer_ids, number_of_purchases, total_reward_points))

# Number of clusters
k = 3

# Initialize clusters with random centers
clusters = {i: {'center': X[np.random.randint(0, X.shape[0]), 1:], 'points': []} for i in range(k)}

def showgraph(X, clusters):
    plt.scatter(X[:, 1], X[:, 2])
    plt.grid(True)
    for i in clusters:
        center = clusters[i]['center']
        plt.scatter(center[0], center[1], marker='*', c='red')
    plt.show()

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx, 1:]  # Ignore CustomerID for clustering
        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
            showgraph(X, clusters)
    return clusters

def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i, 1:], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

# Run the K-Means algorithm
for _ in range(10):  # Iterate 10 times
    clusters = assign_clusters(X, clusters)
    clusters = update_clusters(X, clusters)

# Final cluster assignment
predictions = pred_cluster(X, clusters)
print(predictions)