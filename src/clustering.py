"""
Module: Clustering Logic
Description: 
    Encapsulates machine learning algorithms for customer segmentation.
    Includes implementations for K-Means Clustering (Elbow Method, fitting)
    and Hierarchical Clustering (Dendrograms, Agglomerative fitting).

Functions:
    - get_inertia_values: Computes WCSS for Elbow Method.
    - fit_kmeans: Trains the K-Means model.
    - get_linkage_matrix: Computes the linkage matrix for hierarchical clustering.
    - perform_hierarchical_clustering: Trains Agglomerative model.
"""

from sklearn.cluster import KMeans
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

def get_inertia_values(data: pd.DataFrame, max_k: int = 10):
    """
    Calculates the inertia (WCSS) for a range of cluster numbers.
    Used for the Elbow Method.
    
    Args:
        data (pd.DataFrame): The scaled data.
        max_k (int): Maximum number of clusters to test.
        
    Returns:
        list: List of inertia values.
    """
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    return inertia

def fit_kmeans(data: pd.DataFrame, n_clusters: int):
    """
    Fits the K-Means algorithm to the provided data.
    
    Args:
        data (pd.DataFrame): Subset of data (numerical features only).
        n_clusters (int): The number of clusters (K).
        
    Returns:
        tuple: (labels, centroids)
            - labels: Array of cluster labels for each data point.
            - centroids: Array of cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    
    labels = kmeans.fit_predict(data)
    
    centroids = kmeans.cluster_centers_
    
    return labels, centroids

def get_linkage_matrix(data: pd.DataFrame, method: str = 'ward'):
    """
    Computes the linkage matrix for hierarchical clustering.
    
    Args:
        data (pd.DataFrame): Scaled data.
        method (str): Linkage method (default 'ward').
        
    Returns:
        np.array: Linkage matrix.
    """
    # Scipy's linkage function handles the hierarchy calculation
    return linkage(data, method=method)

def perform_hierarchical_clustering(data: pd.DataFrame, n_clusters: int):
    """
    Fits Agglomerative Hierarchical Clustering.
    
    Args:
        data (pd.DataFrame): Scaled data.
        n_clusters (int): Number of clusters to find.
        
    Returns:
        np.array: Cluster labels.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    return model.fit_predict(data)