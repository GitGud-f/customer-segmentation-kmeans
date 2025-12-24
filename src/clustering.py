"""
Module: Clustering Logic
Description: 
    Encapsulates machine learning algorithms for customer segmentation.
    Includes implementations for K-Means Clustering (Elbow Method, fitting)
    and Hierarchical Clustering (Dendrograms, Agglomerative fitting).

Functions:
    - get_inertia_values: Computes WCSS for Elbow Method.
    - fit_kmeans: Trains the K-Means model.
"""

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

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
    # Initialize KMeans with k-means++ for better initialization
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    
    # Fit the model and predict labels
    labels = kmeans.fit_predict(data)
    
    # Get the coordinates of the cluster centers
    centroids = kmeans.cluster_centers_
    
    return labels, centroids