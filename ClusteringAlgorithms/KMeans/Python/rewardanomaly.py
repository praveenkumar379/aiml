import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data (replace with your actual loading method)
# try:
data = pd.read_csv("rewards_data.csv") # if you exported to CSV
# except:
#     # Or connect directly to your database here if needed. Example using SQLAlchemy:
#     from sqlalchemy import create_engine
#     engine = create_engine('your_database_connection_string') #e.g., 'postgresql://user:password@host:port/database'
#     data = pd.read_sql_table('YourTableName', engine)

# Select the features for clustering
features = ['NewRewardPercentage', 'RenewRewardPercentage']
X = data[features]

# Scale the data (important for k-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters (k) - Elbow Method
inertia = []
for k in range(1, 11):  # Try k values from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init helps avoid issues with random initialization
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose the optimal k based on the Elbow Method and re-run the k-means algorithm
optimal_k = 3  # Replace with the k value you determine from the Elbow Method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze the clusters
for i in range(optimal_k):
    cluster_data = data[data['cluster'] == i]
    print(f"Cluster {i}:")
    print(cluster_data[features].describe())  # Descriptive statistics for each cluster
    # Further analysis: look at the CustomerId, ProductID, dates, etc.
    print("-" * 50)

# Identify potential anomalies (examples)
# Look for clusters with extreme values or unexpected combinations
anomaly_clusters = data[data['NewRewardPercentage'] > 0.5] # Example:  New reward over 50%
print("\nPotential Anomalies:")
print(anomaly_clusters)

# Or analyze based on distance to centroid:
from sklearn.metrics import pairwise_distances_argmin_min
centroids = kmeans.cluster_centers_
distances, indices = pairwise_distances_argmin_min(X_scaled, centroids)
data['distance_to_centroid'] = distances
anomalies_distance = data[data['distance_to_centroid'] > 1] # Example threshold
print("\nAnomalies based on distance to centroid:")
print(anomalies_distance)


# Visualize the clusters (optional but helpful)
plt.scatter(X['NewRewardPercentage'], X['RenewRewardPercentage'], c=data['cluster'], cmap='viridis')
plt.xlabel('New Reward Percentage')
plt.ylabel('Renew Reward Percentage')
plt.title('K-means Clustering of Reward Percentages')
plt.show()