# Step 1: Importing Necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Loading the Dataset

try:
   
    insurance_df = pd.read_csv('insurance.csv')
    print("Dataset loaded successfully!")

except FileNotFoundError:
    print("Error: 'insurance.csv' not found. Please check the file path.")

    insurance_df = pd.DataFrame()

# Step 3: Initial Data Inspection

if not insurance_df.empty:
    # To get a quick overview.
    print("\nFirst 5 rows of the dataset:")
    print(insurance_df.head())

    # Dimensions of the dataset.
    print(f"\nDataset shape: {insurance_df.shape} rows and {insurance_df.shape[1]} columns.")

    # Concise summary of the DataFrame.
    print("\nDataset information:")
    insurance_df.info()

    # Check for any missing values in each column.
    print("\nMissing values in each column:")
    print(insurance_df.isnull().sum())

    # Descriptive statistics for the numerical columns (like mean, min, max, etc.).
    print("\nDescriptive statistics for numerical columns:")
    print(insurance_df.describe())

# Step 4: Exploratory Data Analysis (EDA) - Visualizing the Data

if not insurance_df.empty:
    print("\nStarting Exploratory Data Analysis...")

 
    sns.set_style('whitegrid')

    # a) Distribution of Charges
    plt.figure(figsize=(12, 6))
    sns.histplot(insurance_df['charges'], kde=True, bins=30)
    plt.title('Distribution of Medical Charges')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.savefig('charges_distribution.png')
    plt.show()

    # b) Relationship between 'smoker' and 'charges'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='smoker', y='charges', data=insurance_df)
    plt.title('Medical Charges by Smoking Status')
    plt.xlabel('Smoker')
    plt.ylabel('Charges')
    plt.savefig('smoker_vs_charges_boxplot.png')
    plt.show()

    # c) Relationship between 'age' and 'charges'
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='age', y='charges', data=insurance_df, hue='smoker')
    plt.title('Medical Charges vs. Age')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.savefig('age_vs_charges_scatterplot.png')
    plt.show()
    
    # d) Relationship between 'bmi' and 'charges'
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='bmi', y='charges', data=insurance_df, hue='smoker')
    plt.title('Medical Charges vs. BMI')
    plt.xlabel('Body Mass Index (BMI)')
    plt.ylabel('Charges')
    plt.savefig('bmi_vs_charges_scatterplot.png')
    plt.show()

    # e) Correlation Heatmap for numerical features
    # First, we select only the numerical columns for the correlation matrix
    numerical_df = insurance_df.select_dtypes(include=np.number)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()
    plt.savefig('correlation_heatmap.png')
    print("\nEDA visualizations complete.")

    # Step 5: Data Preprocessing - Preparing Data for the Model

if not insurance_df.empty:
    print("Original dataset columns:")
    print(insurance_df.columns)
    print("\nFirst 5 rows of original data:")
    print(insurance_df.head())

    # Converting categorical variables into numerical format using one-hot encoding.
    processed_df = pd.get_dummies(insurance_df, columns=['sex', 'smoker', 'region'], drop_first=True)

    print("\n\nDataset after one-hot encoding:")
    print("Notice the new columns like 'sex_male', 'smoker_yes', etc.")
    print("\nProcessed dataset columns:")
    print(processed_df.columns)
    print("\nFirst 5 rows of processed data:")
    print(processed_df.head())

else:
    print("DataFrame is empty. Please load the data first.")

# Step 6: Building and Training the Linear Regression Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



if 'processed_df' in locals() and not processed_df.empty:
    # a) Separating the features (X) and the target variable (y)
    X = processed_df.drop('charges', axis=1)
    y = processed_df['charges']

    # b) Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # c) Creating and training the Linear Regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    print("\nModel training complete.")

    # d) Making predictions on the test data
    y_pred = lin_reg.predict(X_test)

    # e) Evaluating the model's performance
    r2_score = metrics.r2_score(y_test, y_pred)
    print(f"\nModel Performance on Test Data:")
    print(f"R-squared (R2): {r2_score:.4f}") 

    # f) Interpreting the model 
    print("\nModel Interpretation:")
    print(f"Intercept: {lin_reg.intercept_:.2f}")
    coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)

else:
    print("Processed DataFrame not found. Please run the preprocessing step first.")

# Step 7: Final Accuracy Summary

print("\n----------------------------------------------------")
print(f"Final Model Accuracy (R-squared): {r2_score * 100:.2f}%")
print("----------------------------------------------------")
print(f"This means the model can explain approximately {r2_score * 100:.0f}% of the variation in medical charges.")