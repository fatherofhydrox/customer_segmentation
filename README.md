# 👥 Customer Segmentation using K-Means Clustering

This project applies unsupervised machine learning to segment customers based on their spending behavior. It helps businesses identify distinct customer groups to improve targeting, personalization, and marketing strategy.

## 📌 Objective

Segment customers into different clusters based on features like **Annual Income** and **Spending Score** using **K-Means Clustering**.

## 📂 Dataset

- Source: [Mall Customer Segmentation Data](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial)
- Features:
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

## 🛠️ Technologies Used

- **Language:** Python
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Algorithm:** K-Means Clustering
- **IDE:** Jupyter Notebook

## 📊 Exploratory Data Analysis (EDA)

- Age, income, and spending behavior distributions
- Gender-based comparisons
- Heatmaps and pairplots for correlation visualization

## 📌 Clustering Process

1. **Feature Selection:** Chose relevant features (`Annual Income`, `Spending Score`)
2. **Scaling:** StandardScaler to normalize data
3. **Finding Optimal Clusters:** Elbow method with Within-Cluster-Sum-of-Squares (WCSS)
4. **Clustering:** Applied `KMeans(n_clusters=5)`
5. **Visualization:** 2D scatter plot with colored clusters

## 📈 Results

- Identified 5 meaningful customer segments:
  - High income & high spenders
  - Low income & high spenders
  - Average income & average spenders
  - etc.
- Visualized clusters using `matplotlib` and `seaborn`
- Helped derive actionable insights for targeted marketing

## 💡 Business Use Case

- Personalized marketing campaigns
- Customer loyalty programs
- Resource allocation based on segment profitability

## 🚀 Future Improvements

- Apply PCA for dimensionality reduction
- Use DBSCAN or Hierarchical Clustering
- Deploy as an interactive Streamlit app

## 📎 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/fatherofhydrox/customer_segmentation
   cd customer_segmentation
pip install -r requirements.txt
jupyter notebook customer_segmentation.ipynb

🙌 Acknowledgements
Inspired by customer behavior analytics and marketing applications.
Dataset from Kaggle.

🚀 **Live App:** [Try it here](https://customer-segmentation01.onrender.com)
