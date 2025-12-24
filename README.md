# Mall Customer Segmentation Analysis

## Project Overview
This project performs a comprehensive analysis of mall customer data to identify distinct customer segments. By leveraging unsupervised machine learning techniques—specifically **K-Means Clustering** and **Hierarchical Agglomerative Clustering**—this analysis derives actionable insights for targeted marketing strategies.

Unlike standard data science notebooks, this project utilizes a **software engineering approach**, separating logic into reusable Python modules (`src/`) to ensure clean, maintainable, and scalable code, while using Jupyter Notebooks (`notebooks/`) strictly for narrative and visualization.

### Key Objectives
1.  **Data Exploration:** Analyze distributions and correlations between Age, Income, and Spending Score.
2.  **Segmentation:** Group customers into clusters using K-Means (Elbow Method) and Hierarchical Clustering (Dendrograms).
3.  **Visualization:** Create interactive 3D visualizations to interpret complex customer behaviors.
4.  **Strategy:** Provide business recommendations based on the identified segments (e.g., "High Income, Low Spenders").

---

## Project Structure

The project follows a modular file structure to separate data, source code, and analysis.

```text
customer-segmentation/
│
├── data/
│   ├── raw/                  # Original immutable dataset
│   │   └── Mall_Customers.csv
│   └── processed/            # Cleaned and scaled data
│       └── Mall_Customers_Clean.csv
│
├── docs/                     # Pdoc generated docuemntation
│
├── notebooks/
│   └── analysis_report.ipynb # Main analysis (The Entry Point)
│
├── src/                      # Source code 
│   ├── __init__.py
│   ├── data_loader.py        # Data loading, cleaning, and scaling logic
│   ├── visualization.py      # Plotting functions (Matplotlib, Seaborn, Plotly)
│   └── clustering.py         # K-Means and Hierarchical Clustering algorithms
│
├── .gitignore                # Files to exclude from Git
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## Dataset Description
The analysis is based on the `Mall_Customers.csv` dataset, which contains data on 200 mall customers.

| Column | Description | Data Type |
| :--- | :--- | :--- |
| **CustomerID** | Unique ID assigned to the customer (Dropped during preprocessing) | Integer |
| **gender** | Gender of the customer | Categorical |
| **age** | Age of the customer | Integer |
| **yearly income** | Annual Income of the customer | Integer |
| **purchase spending** | Score assigned by the mall based on customer behavior and spending nature | Integer |

---

## Installation & Setup

Follow these steps to set up the environment and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/k3rnel-paN1c5/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Register Kernel for Jupyter
To ensure Jupyter uses the virtual environment we just created:
```bash
python -m ipykernel install --user --name=mall-segmentation-env --display-name "Python (Mall Segmentation)"
```

---

## How to Run the Analysis

1.  Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
2.  Open the file: `notebooks/analysis_report.ipynb`.
3.  Run all cells to execute the data pipeline, generate plots, and view the 3D clusters.

---

## Methodology & Insights

### 1. Data Preprocessing
*   Removed `CustomerID` (non-informative).
*   Checked for missing values and duplicates.
*   Scaled features using `MinMaxScaler` for distance-based algorithms.

### 2. K-Means Clustering
*   Used the **Elbow Method** to determine the optimal number of clusters (K=6).
*   Visualized clusters in 2D (income vs. spending) and 3D (age vs. income vs. spending).

### 3. Hierarchical Clustering 
*   Generated a **Dendrogram** using Ward's linkage to confirm the optimal cluster count.
*   Compared Agglomerative Clustering results with K-Means.

### 4. Key Segments Identified
The analysis identified several distinct customer groups, including:
*   **"Target Customers" (Whales):** High Income, High Spending Score. *Strategy: VIP offers and luxury branding.*
*   **"Sensible Savers":** High Income, Low Spending Score. *Strategy: Value-based promotions to unlock potential.*
*   **"Careless Spenders":** Low Income, High Spending Score. *Strategy: Discount coupons and app-based rewards.*

---

## 🛠️ Technologies Used
*   **Language:** Python 3.x
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn (KMeans, AgglomerativeClustering)
*   **Visualization:** Matplotlib, Seaborn, Plotly (Interactive 3D)
*   **Tools:** Jupyter Notebook, Git

---

*This project was completed as part of the Data Mining Course curriculum.*
```