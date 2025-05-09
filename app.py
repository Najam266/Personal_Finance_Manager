import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback

# Step 1: Load the datasets
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_datasets():
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct paths relative to the current directory
        budget_path = os.path.join(current_dir, "Preprocessed Datasets", "Budget1.csv")
        transactions_path = os.path.join(current_dir, "Preprocessed Datasets", "personal_transactions1.csv")
        monthly_summary_path = os.path.join(current_dir, "Preprocessed Datasets", "monthly_summary1.csv")
        investment_path = os.path.join(current_dir, "Preprocessed Datasets", "investment_portfolio1.csv")
        
        print(f"Loading budget from: {budget_path}")
        budget = pd.read_csv(budget_path)
        print("Budget loaded successfully")
        
        print(f"Loading transactions from: {transactions_path}")
        transactions = pd.read_csv(transactions_path)
        print("Transactions loaded successfully")
        
        print(f"Loading monthly summary from: {monthly_summary_path}")
        monthly_summary = pd.read_csv(monthly_summary_path)
        print("Monthly summary loaded successfully")
        
        print(f"Loading investment from: {investment_path}")
        investment = pd.read_csv(investment_path)
        print("Investment loaded successfully")

        print("✅ Files loaded successfully!")
        print("Budget:", budget.shape)
        print("Transactions:", transactions.shape)
        print("Monthly Summary:", monthly_summary.shape)
        print("Investment:", investment.shape)

        return budget, transactions, monthly_summary, investment

    except FileNotFoundError as e:
        print(f"❌ File not found: {str(e)}")
        return None, None, None, None
    except pd.errors.EmptyDataError as e:
        print(f"❌ Empty dataset: {str(e)}")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Error loading datasets: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return None, None, None, None

# Step 2: Feature Engineering
@st.cache_data(ttl=3600)  # Cache for 1 hour
def engineer_features(budget, transactions, monthly_summary, investment):
    try:
        # Transactions
        transactions['Date'] = pd.to_datetime(transactions['Date'], format='%m/%d/%Y', errors='coerce')
        transactions['Month'] = transactions['Date'].dt.month
        transactions['DayOfWeek'] = transactions['Date'].dt.day_name()
        transactions['CumulativeAmount'] = transactions.groupby('Month')['Amount'].cumsum()

        # Monthly Summary
        # Convert Month to datetime and extract month number
        monthly_summary['Month'] = pd.to_datetime(monthly_summary['Month'], format='%Y-%m').dt.month
        # Convert monetary columns to float
        monetary_columns = ['Income', 'Expenses', 'Savings']
        for col in monetary_columns:
            if monthly_summary[col].dtype == 'object':
                monthly_summary[col] = monthly_summary[col].str.replace('$', '').str.replace(',', '').astype(float)
        monthly_summary['SavingsRate'] = monthly_summary['Savings'] / (monthly_summary['Income'] + 1e-5)

        # Budget
        # Convert Budget to float if it's not already
        if budget['Budget'].dtype == 'object':
            budget['Budget'] = budget['Budget'].str.replace('$', '').str.replace(',', '').astype(float)

        # Investment
        investment['Date'] = pd.to_datetime(investment['Date_of_Investment'], format='%d/%m/%Y', errors='coerce')
        investment['Month'] = investment['Date'].dt.month
        # Convert monetary columns to float
        monetary_columns = ['Current_Value', 'Amount_Invested']
        for col in monetary_columns:
            if investment[col].dtype == 'object':
                investment[col] = investment[col].str.replace('$', '').str.replace(',', '').astype(float)
        investment['InvestmentValueChange'] = investment['Current_Value'] - investment['Amount_Invested']

        print("✅ Feature engineering completed.")
        return budget, transactions, monthly_summary, investment
    except Exception as e:
        print(f"❌ Error in feature engineering: {str(e)}")
        print("Traceback:", traceback.format_exc())
        raise e

# Step 3: Data Preprocessing
@st.cache_data(ttl=3600)  # Cache for 1 hour
def preprocess_data(budget, transactions, monthly_summary, investment):
    try:
        # Ensure all monetary columns are already float
        monetary_columns = ['Income', 'Expenses', 'Savings']
        for col in monetary_columns:
            if monthly_summary[col].dtype == 'object':
                monthly_summary[col] = monthly_summary[col].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Create a monthly budget by replicating the budget for each month
        months = monthly_summary['Month'].unique()
        budget_data = []
        for month in months:
            for _, row in budget.iterrows():
                budget_data.append({
                    'Month': month,
                    'Category': row['Category'],
                    'Budgeted': row['Budget']
                })
        budget_df = pd.DataFrame(budget_data)
        
        # Combine relevant features for prediction
        combined_data = pd.merge(
            monthly_summary[['Month', 'Income', 'Expenses', 'Savings']],
            budget_df,
            on='Month',
            how='left'
        )
        
        # Handle categorical variables
        le = LabelEncoder()
        combined_data['Category'] = le.fit_transform(combined_data['Category'])
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['Income', 'Expenses', 'Savings', 'Budgeted']
        combined_data[numerical_cols] = scaler.fit_transform(combined_data[numerical_cols])
        
        print("✅ Data preprocessing completed.")
        return combined_data, scaler, le
    except Exception as e:
        print(f"❌ Error in data preprocessing: {str(e)}")
        print("Traceback:", traceback.format_exc())
        raise e

# Step 4: Model Training and Evaluation
@st.cache_data(ttl=3600)  # Cache for 1 hour
def train_and_evaluate_models(data):
    try:
        # Prepare features and target
        X = data.drop(['Savings'], axis=1)
        y = data['Savings']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42)
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'metrics': {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }
            }
            
            print(f"\n{name} Results:")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R2: {r2:.2f}")
        
        return results
    except Exception as e:
        print(f"❌ Error in model training: {str(e)}")
        print("Traceback:", traceback.format_exc())
        raise e

def main():
    try:
        st.set_page_config(page_title="Personal Finance Manager", layout="wide")
        st.title("Personal Finance Manager")
        
        # Load data
        with st.spinner("Loading data..."):
            budget, transactions, monthly_summary, investment = load_datasets()
            
        if budget is None:
            st.error("Failed to load data. Please check the console for errors.")
            return
            
        # Process data
        with st.spinner("Processing data..."):
            budget, transactions, monthly_summary, investment = engineer_features(
                budget, transactions, monthly_summary, investment
            )
            
            combined_data, scaler, le = preprocess_data(
                budget, transactions, monthly_summary, investment
            )
            
            results = train_and_evaluate_models(combined_data)
            
        # Sidebar
        st.sidebar.header("Navigation")
        page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Predictions", "Analysis"])
        
        if page == "Dashboard":
            st.header("Financial Dashboard")
            
            # Convert monetary columns to numerical values if they aren't already
            monetary_columns = ['Income', 'Expenses', 'Savings']
            for col in monetary_columns:
                if monthly_summary[col].dtype == 'object':
                    monthly_summary[col] = monthly_summary[col].str.replace('$', '').str.replace(',', '').astype(float)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Income", f"${monthly_summary['Income'].sum():,.2f}")
            with col2:
                st.metric("Total Expenses", f"${monthly_summary['Expenses'].sum():,.2f}")
            with col3:
                st.metric("Total Savings", f"${monthly_summary['Savings'].sum():,.2f}")
            
            # Plot monthly trends
            st.subheader("Monthly Trends")
            fig, ax = plt.subplots(figsize=(10, 6))
            monthly_summary.plot(x='Month', y=['Income', 'Expenses', 'Savings'], ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        elif page == "Predictions":
            st.header("Savings Predictions")
            
            # Display model performance
            st.subheader("Model Performance")
            for model_name, result in results.items():
                st.write(f"### {model_name}")
                metrics = result['metrics']
                st.write(f"MAE: ${metrics['MAE']:,.2f}")
                st.write(f"RMSE: ${metrics['RMSE']:,.2f}")
                st.write(f"R2: {metrics['R2']:.2f}")
                
        elif page == "Analysis":
            st.header("Financial Analysis")
            
            # Category-wise expense analysis
            st.subheader("Category-wise Expenses")
            category_expenses = budget.groupby('Category')['Budget'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            category_expenses.plot(kind='bar', ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Full traceback:")
        st.code(traceback.format_exc())
        print("Error:", str(e))
        print("Traceback:", traceback.format_exc())

if __name__ == "__main__":
    main() 