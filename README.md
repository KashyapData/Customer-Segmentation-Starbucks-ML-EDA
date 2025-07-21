# 🎯 Customer Segmentation for Starbucks

This project demonstrates a complete end-to-end machine learning pipeline for customer segmentation using data from a simulated Starbucks campaign. It was conducted as a Master’s dissertation and explores how businesses like Starbucks can enhance marketing strategies through data-driven insights.

---

## 📌 Project Objectives

- Segment customers based on demographics, transactions, and campaign responses.
- Use EDA to identify trends and patterns.
- Apply K-Means Clustering to define behavioral segments.
- Validate clusters with a Random Forest classifier.
- Recommend business strategies for each customer segment.

---

## 🔧 Tools & Technologies Used

| Tool/Library      | Purpose                          |
|------------------|----------------------------------|
| Python           | Core programming language        |
| Jupyter Notebook | Development environment          |
| Pandas, NumPy    | Data wrangling & manipulation    |
| Matplotlib, Seaborn | Data visualization             |
| Scikit-learn     | Clustering, Classification        |

---

## 📊 Dataset Overview

- **Profile Dataset**: Demographic data (age, gender, income, membership date)
- **Portfolio Dataset**: Offer type, difficulty, duration, reward
- **Transcript Dataset**: Customer-event interactions (view, receive, complete)

---

## 📈 Methods & Process

### 1. Data Wrangling
- Cleaned and merged three datasets
- Handled missing values and standardized formats

### 2. Exploratory Data Analysis (EDA)
- Univariate and bivariate analysis (age, income, gender)
- Offer view vs completion rates
- Seasonal/temporal membership trends

### 3. Clustering (K-Means)
- Feature scaling with `StandardScaler`
- Optimal cluster count using Elbow and Silhouette methods
- Defined 5 customer segments:
  - 🏆 Most Valuable Customers
  - ☕ Regulars
  - 📈 High Potentials
  - 👀 Offer Viewers
  - 💤 Least Engagers

### 4. Classification (Random Forest)
- Verified clusters using a supervised model
- Achieved >98% accuracy in classification of cluster behavior

---

## 📌 Key Insights

- High-income customers engage better with offers
- Age groups 36–55 showed highest loyalty response
- Time-sensitive promotions perform better midweek
- Personalization boosts conversion by 2x in targeted groups
