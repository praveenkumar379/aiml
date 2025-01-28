import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

samples = 1500
seed = 50
centers = 3
dimentions = 2
clusterCount = 3

X,y = make_blobs(n_samples = samples,n_features = dimentions,centers = centers,random_state = seed, shuffle=True)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1])
plt.show()

# print(X)

k = clusterCount

clusters = {}
np.random.seed(seed)

for idx in range(k):
    center = 2*(2*np.random.random((X.shape[1],))-1)
    points = []
    cluster = {
        'center' : center,
        'points' : []
    }
    
    clusters[idx] = cluster
    
# print(clusters)

def showgraph(X,clusters):
    plt.scatter(X[:,0],X[:,1])
    plt.grid(True)
    for i in clusters:
        center = clusters[i]['center']
        plt.scatter(center[0],center[1],marker = '*',c = 'red')
    plt.show()

def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        
        curr_x = X[idx]
        
        for i in range(k):
            dis = distance(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)        
    
    return clusters

def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            clusters[i]['center'] = new_center
            
            clusters[i]['points'] = []
            showgraph(X,clusters)
    return clusters

def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred   

showgraph(X,clusters)
    
clusters = assign_clusters(X,clusters)
clusters = update_clusters(X,clusters)
pred = pred_cluster(X,clusters)

plt.scatter(X[:,0],X[:,1],c = pred)
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker = 'x',c = 'red')
plt.show()
