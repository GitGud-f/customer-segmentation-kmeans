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
"""

import matplotlib.pyplot as plt
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