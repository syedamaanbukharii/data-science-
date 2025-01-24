
# coding: utf-8

# In[29]:


# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset (replace the file path with your actual file)
data = pd.read_csv('10_births_hourly_data.csv')

# Inspect the dataset (first few rows)
print(data.head())

# Data cleaning and preprocessing
# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Check for missing values
print(data.isnull().sum())

# Fill missing values with median or another strategy (if necessary)
data.fillna(data.median(), inplace=True)

# Check the data types
print(data.dtypes)

# Convert time to datetime format (if necessary)
data['Time (hours)'] = pd.to_datetime(data['Time (hours)'])
data.columns = data.columns.str.strip()

# Let's visualize the distribution of cervical dilation
plt.figure(figsize=(8, 6))
sns.distplot(data['Cervical Dilatation (cm)'], kde=True, color="purple")
plt.title('Distribution of Cervical Dilation')
plt.xlabel('Cervical Dilatation (cm)')
plt.ylabel('Frequency')
plt.show()

# Create a time vs cervical dilation plot (partograph-like graph)
plt.figure(figsize=(10, 6))
#sns.tsplot(x='Time', y='Cervical_Dilation', data=data, marker='o', color='blue')
plt.title('Labor Progress - Cervical Dilation Over Time (Partograph)')
plt.xlabel('Time (hours)')
plt.ylabel('Cervical Dilation (cm)')
plt.xticks(rotation=45)  # Rotate time labels if needed
plt.tight_layout()
plt.show()

# Heatmap of cervical dilation vs time (for hourly analysis)
data['Hour'] = data['Time (hours)'].dt.hour
heatmap_data = data.pivot_table(values='Cervical Dilatation (cm)', index='Hour', columns='Case ID', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=False)
plt.title('Heatmap of Cervical Dilatation by Hour and Patient')
plt.xlabel('Case ID')
plt.ylabel('Hour of Labor')
plt.tight_layout()
plt.show()

# Pairplot to explore relationships between different columns (if more columns exist)
sns.pairplot(data)
plt.show()

# Using sklearn for scaling data (if necessary for certain models or analysis)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Cervical Dilatation (cm)', 'Hour']])  # Scale relevant columns
scaled_df = pd.DataFrame(scaled_data, columns=['Cervical Dilatation (cm)', 'Hour'])
print(scaled_df.head())

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Variables')
plt.show()

# A potential classification task: predict if labor is progressing normally (assuming some target column like 'RiskLevel')
# Example of a simple classifier using sklearn:

# Assuming 'RiskLevel' is a target column (with values like 'Normal', 'Risky')
if 'RiskLevel' in data.columns:
    X = data[['Cervical Dilatation (cm)', 'Hour']]  # Features
    y = data['RiskLevel']  # Target

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Evaluate the model
    print(f"Model Accuracy: {clf.score(X_test, y_test):.2f}")
else:
    print("RiskLevel column is not available for classification.")

