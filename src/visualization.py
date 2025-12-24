"""
Module: Visualization
Description: 
    Contains reusable plotting functions for Exploratory Data Analysis (EDA).
    Supports univariate, bivariate, and multivariate analysis using Matplotlib
    and Seaborn. Also includes functions for cluster visualization.

Functions:
    - plot_gender_distribution: Bar chart for categorical data.
    - plot_histograms: Distribution plots for numerical features.
    - plot_bivariate_scatter: Scatter plots for feature pairs.
    - plot_multivariate_bubble: 2D scatter with point size representing a 3rd dimension.
    - plot_kmeans_clusters: Visualizes K-Means clustering results with centroids.
    - plot_elbow_curve: Plots the Elbow Method graph.
    - plot_3d_static: Static 3D scatter plot using Matplotlib.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

# Set global styles for plots
sns.set_style("whitegrid")
sns.set_palette("muted")

def plot_gender_distribution(df: pd.DataFrame, column: str = 'gender'):
    """
    Creates a bar chart showing the percentage distribution of Gender.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        column (str): Column name for gender
    """
    plt.figure(figsize=(6, 4))
    
    # Calculate percentages
    counts = df[column].value_counts(normalize=True) * 100
    
    # Plot
    ax = sns.barplot(x=counts.index, y=counts.values)
    
    # Add labels
    plt.title(f'Percentage Distribution of {column}', fontsize=14)
    plt.ylabel('Percentage (%)')
    plt.xlabel(column)
    
    # Add text annotations on bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.5,
                f'{height:.1f}%', ha="center")
    
    plt.show()

def plot_histograms(df: pd.DataFrame, columns: list):
    """
    Creates distribution plots (histograms with KDE) for a list of columns.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        columns (list): List of column names to plot.
    """
    plt.figure(figsize=(15, 5))
    
    for i, col in enumerate(columns):
        plt.subplot(1, 3, i + 1)
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
    plt.tight_layout()
    plt.show()
    

def plot_bivariate_scatter(df: pd.DataFrame, pairs: list):
    """
    Creates scatter plots for specific pairs of features.
    
    Args:
        pairs (list of tuples): e.g., [('age', 'yearly income (k$)'), ...]
    """
    plt.figure(figsize=(15, 5))
    
    for i, (x_col, y_col) in enumerate(pairs):
        plt.subplot(1, 3, i + 1)
        sns.scatterplot(data=df, x=x_col, y=y_col, s=60, alpha=0.7)
        plt.title(f'{x_col} vs {y_col}')
        
    plt.tight_layout()
    plt.show()

def plot_multivariate_bubble(df: pd.DataFrame, x_col: str, y_col: str, size_col: str):
    """
    Creates a 2D scatter plot using a 3rd variable for point size (Bubble Plot).
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        size_col (str): Column name for point size.
    """
    
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        size=size_col, 
        sizes=(20, 200), # Min and Max size of bubbles
        hue=size_col,    # color by age too for clarity
        palette="viridis",
        alpha=0.7,
        legend='brief'
    )
    
    plt.title(f'{x_col} vs {y_col} (Size = {size_col})', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) 
    plt.show()
    
def plot_kmeans_clusters(df: pd.DataFrame, x_col: str, y_col: str, labels, centroids):
    """
    Visualizes K-Means clustering results (2D) with centroids.
    
    Args:
        df (pd.DataFrame): The dataframe containing the features.
        x_col (str): Name of the X-axis feature.
        y_col (str): Name of the Y-axis feature.
        labels (array): Cluster labels.
        centroids (array): Coordinates of cluster centers.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the data points colored by cluster
    sns.scatterplot(
        x=df[x_col], 
        y=df[y_col], 
        hue=labels, 
        palette='viridis', 
        s=100, 
        alpha=0.8,
        legend='full'
    )
    
    # Plot the centroids
    # centroids[:, 0] is x-coordinates, centroids[:, 1] is y-coordinates
    plt.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        s=300, 
        c='red', 
        marker='X', 
        label='Centroids'
    )
    
    plt.title(f'K-Means Clustering: {x_col} vs {y_col}', fontsize=15)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()
    
def plot_elbow_curve(inertia_values: list):
    """
    Plots the Elbow Method graph.
    Args: 
        inertia_values (list): List of inertia (WCSS) values for different K.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(inertia_values) + 1), inertia_values, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.xticks(range(1, len(inertia_values) + 1))
    plt.show()
    
def plot_3d_static(df: pd.DataFrame, x_col, y_col, z_col, labels):
    """
    Creates a static 3D scatter plot using Matplotlib.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        z_col (str): Column name for z-axis.
        labels (array): Cluster labels for coloring points.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df[x_col], 
        df[y_col], 
        df[z_col], 
        c=labels, 
        cmap='viridis', 
        s=60
    )
    
    ax.set_title("3D Customer Segments")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    
    plt.colorbar(scatter, pad=0.1)
    plt.show()