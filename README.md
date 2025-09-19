# Assignment 1: Predicting Medical Insurance Costs

## 1. Project Goal

The goal of this project is to understand what factors influence medical insurance costs and to build a model that can predict them. Using a public dataset of insurance beneficiaries, this analysis explores how personal details like age, BMI, and smoking habits affect a person's medical charges.

The project covers the complete data analysis process:

*   Loading and understanding the data
*   Finding patterns with data visualization
*   Preparing the data for a machine learning model
*   Building and training a Linear Regression model
*   Evaluating how well the model performs

-----

## 2. About the Dataset

This project uses the **Medical Cost Personal Dataset** from Kaggle. It contains 1,338 records of individuals with the following information:

*   `age`: The person's age.
*   `sex`: Their gender.
*   `bmi`: Body Mass Index.
*   `children`: Number of children they have.
*   `smoker`: Whether they smoke or not.
*   `region`: Their residential area in the US.
*   `charges`: The medical costs billed by their insurance.

**Dataset Source:** [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

-----

## 3. How It Works

The analysis was done in a few key steps:

1.  **Exploratory Data Analysis (EDA):** I created charts and plots to see the data visually. This helped to quickly find important patterns, like the fact that smokers have much higher medical costs.
2.  **Data Preparation:** The model needs all data to be in a numerical format, so I converted text-based columns (like `smoker` and `region`) into numbers using a technique called one-hot encoding.
3.  **Model Training:** I split the data into a training set (80%) and a testing set (20%). The Linear Regression model learned from the training data to find the relationship between the features and the medical charges.
4.  **Evaluation:** Finally, I tested the model on the data it had never seen before (the testing set) to check its performance.

-----

## 4. Results

The model performed well, achieving a final **R-squared score of 78.36%**.

In simple terms, this means the model can explain about **78%** of the variation in medical charges, which is a strong result. The analysis confirmed that **smoking status** is the single biggest factor driving up medical costs.
