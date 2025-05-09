import streamlit as st
st.set_page_config(page_title="Personal Finance Manager", layout="wide")

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
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime
import time
from sklearn.cluster import KMeans
import plotly.express as px

# Configure Gemini API
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    model_name = "gemini-1.5-flash"
    model = genai.GenerativeModel(model_name)
    GEMINI_AVAILABLE = True
except Exception as e:
    GEMINI_AVAILABLE = False
    model = None

# Add custom CSS for enhanced UI
def load_css():
    st.markdown("""
    <style>
    /* Main theme colors and styles */
    :root {
        --primary-color: #2962FF;
        --secondary-color: #82B1FF;
        --background-color: #1E1E1E;
        --card-background: #2D2D2D;
        --text-color: #FFFFFF;
        --text-secondary: #B0B0B0;
        --success-color: #00C853;
        --warning-color: #FFD600;
        --danger-color: #FF1744;
        --border-color: #404040;
    }
    
    /* Global styles */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--card-background);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--card-background);
    }
    
    /* Card styles */
    .stCard {
        background-color: var(--card-background);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        margin: 15px 0;
        border: 1px solid var(--border-color);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2962FF, #82B1FF);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: none;
    }
    
    /* Button styles */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        background: linear-gradient(135deg, #1E88E5, #64B5F6);
    }
    
    /* Form styles */
    .stForm {
        background-color: var(--card-background);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid var(--border-color);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        background-color: #3D3D3D;
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Select boxes */
    .stSelectbox>div>div {
        background-color: #3D3D3D;
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Chart containers */
    .chart-container {
        background-color: var(--card-background);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 15px 0;
        border: 1px solid var(--border-color);
    }
    
    /* Success and warning messages */
    .success-message {
        background: linear-gradient(135deg, #00C853, #69F0AE);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .warning-message {
        background: linear-gradient(135deg, #FFD600, #FFEA00);
        color: #000;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color);
        font-weight: 600;
    }
    
    /* Text */
    p, .stMarkdown {
        color: var(--text-secondary);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--card-background);
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: var(--card-background);
        color: var(--text-color);
    }
    
    /* Metric value styling */
    .stMetric {
        background-color: var(--card-background);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
    }
    
    .stMetric label {
        color: var(--text-secondary);
    }
    
    .stMetric div {
        color: var(--text-color);
    }
    
    /* Plotly chart styling */
    .js-plotly-plot {
        background-color: var(--card-background) !important;
    }
    
    /* Custom title styling */
    .centered-title {
        text-align: center;
        font-size: 2.5em;
        background: linear-gradient(135deg, #2962FF, #82B1FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1em 0;
        text-shadow: none;
        font-weight: bold;
    }
    
    /* Navigation Bar Styles */
    .nav-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem 0;
        margin: 1rem 0;
        background-color: var(--card-background);
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
    }
    
    .nav-item {
        padding: 0.5rem 1.5rem;
        margin: 0 0.5rem;
        color: var(--text-secondary);
        text-decoration: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
        cursor: pointer;
        background: none;
        border: none;
        font-size: 1rem;
    }
    
    .nav-item:hover {
        background-color: var(--primary-color);
        color: white;
        transform: translateY(-2px);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide the default sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .sidebar .sidebar-content {display: none;}
    
    /* Navigation Button Styles */
    div[data-testid="stButton"] > button {
        background-color: #1a237e !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 6px 15px !important;
        margin: 0 2px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        min-width: 100px !important;
        height: 32px !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-testid="stButton"] > button:hover {
        background-color: #283593 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Active button style */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #3949ab !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Navigation container style */
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stButton"]) {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 4px !important;
        padding: 8px !important;
        background-color: #1a237e !important;
        border-radius: 4px !important;
        margin: 10px 0 !important;
    }
    
    /* Make buttons inline */
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stButton"]) {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
    }
    
    /* Remove any extra padding or margins */
    .main .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Welcome message function
def show_welcome_message():
    # Create a placeholder for the welcome message
    welcome_placeholder = st.empty()
    
    # Show welcome message
    welcome_placeholder.markdown("""
    <style>
    .welcome-message {
        font-size: 2.5em;
        text-align: center;
        color: #1E88E5;
        margin: 2em 0;
    }
    </style>
    <div class="welcome-message">
        Welcome to Personal Finance Manager
    </div>
    """, unsafe_allow_html=True)
    
    # Wait for 3 seconds
    time.sleep(3)
    
    # Clear the welcome message
    welcome_placeholder.empty()

def show_centered_title():
    st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 2.5em;
        color: #1E88E5;
        margin: 1em 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: bold;
    }
    </style>
    <div class="centered-title">
        Personal Finance Manager
    </div>
    """, unsafe_allow_html=True)

# Step 1: Load the datasets
@st.cache_data(ttl=3600)
def load_datasets():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        budget_path = os.path.join(current_dir, "Preprocessed Datasets", "Budget1.csv")
        transactions_path = os.path.join(current_dir, "Preprocessed Datasets", "personal_transactions1.csv")
        monthly_summary_path = os.path.join(current_dir, "Preprocessed Datasets", "monthly_summary1.csv")
        investment_path = os.path.join(current_dir, "Preprocessed Datasets", "investment_portfolio1.csv")

        # Check if files exist
        for path in [budget_path, transactions_path, monthly_summary_path, investment_path]:
            if not os.path.exists(path):
                st.error(f"File not found: {path}")
                return None, None, None, None

        # Load datasets with error handling for each file
        try:
            budget = pd.read_csv(budget_path)
        except Exception as e:
            st.error(f"Error loading budget: {str(e)}")
            return None, None, None, None

        try:
            transactions = pd.read_csv(transactions_path)
        except Exception as e:
            st.error(f"Error loading transactions: {str(e)}")
            return None, None, None, None

        try:
            monthly_summary = pd.read_csv(monthly_summary_path)
        except Exception as e:
            st.error(f"Error loading monthly summary: {str(e)}")
            return None, None, None, None

        try:
            investment = pd.read_csv(investment_path)
        except Exception as e:
            st.error(f"Error loading investment data: {str(e)}")
            return None, None, None, None

        return budget, transactions, monthly_summary, investment

    except Exception as e:
        st.error(f"Unexpected error in load_datasets: {str(e)}")
        return None, None, None, None

# Step 2: Feature Engineering
@st.cache_data(ttl=3600)
def engineer_features(budget, transactions, monthly_summary, investment):
    try:
        transactions['Date'] = pd.to_datetime(transactions['Date'], format='%m/%d/%Y', errors='coerce')
        transactions['Month'] = transactions['Date'].dt.month
        transactions['DayOfWeek'] = transactions['Date'].dt.day_name()
        transactions['CumulativeAmount'] = transactions.groupby('Month')['Amount'].cumsum()

        monthly_summary['Month'] = pd.to_datetime(monthly_summary['Month'], format='%Y-%m').dt.month
        for col in ['Income', 'Expenses', 'Savings']:
            if monthly_summary[col].dtype == 'object':
                monthly_summary[col] = monthly_summary[col].str.replace('$', '').str.replace(',', '').astype(float)
        monthly_summary['SavingsRate'] = monthly_summary['Savings'] / (monthly_summary['Income'] + 1e-5)

        if budget['Budget'].dtype == 'object':
            budget['Budget'] = budget['Budget'].str.replace('$', '').str.replace(',', '').astype(float)

        investment['Date'] = pd.to_datetime(investment['Date_of_Investment'], format='%d/%m/%Y', errors='coerce')
        investment['Month'] = investment['Date'].dt.month
        for col in ['Current_Value', 'Amount_Invested']:
            if investment[col].dtype == 'object':
                investment[col] = investment[col].str.replace('$', '').str.replace(',', '').astype(float)
        investment['InvestmentValueChange'] = investment['Current_Value'] - investment['Amount_Invested']

        return budget, transactions, monthly_summary, investment
    except Exception as e:
        print("❌ Error in feature engineering:", traceback.format_exc())
        raise e

# Step 3: Data Preprocessing
@st.cache_data(ttl=3600)
def preprocess_data(budget, transactions, monthly_summary, investment):
    try:
        for col in ['Income', 'Expenses', 'Savings']:
            if monthly_summary[col].dtype == 'object':
                monthly_summary[col] = monthly_summary[col].str.replace('$', '').str.replace(',', '').astype(float)
        
        months = monthly_summary['Month'].unique()
        budget_data = []
        for month in months:
            for _, row in budget.iterrows():
                budget_data.append({'Month': month, 'Category': row['Category'], 'Budgeted': row['Budget']})
        budget_df = pd.DataFrame(budget_data)
        
        combined_data = pd.merge(
            monthly_summary[['Month', 'Income', 'Expenses', 'Savings']],
            budget_df,
            on='Month',
            how='left'
        )
        
        le = LabelEncoder()
        combined_data['Category'] = le.fit_transform(combined_data['Category'])
        
        # Define the feature columns in the correct order (excluding Savings)
        feature_columns = ['Month', 'Income', 'Expenses', 'Budgeted', 'Category']
        
        # Scale only the numerical features (excluding Savings)
        scaler = StandardScaler()
        numerical_cols = ['Income', 'Expenses', 'Budgeted']
        combined_data[numerical_cols] = scaler.fit_transform(combined_data[numerical_cols])
        
        # Store the feature columns order in session state
        st.session_state.feature_columns = feature_columns
        
        return combined_data, scaler, le
    except Exception as e:
        print("❌ Error in data preprocessing:", traceback.format_exc())
        raise e

# Step 4: Model Training and Evaluation
@st.cache_data(ttl=3600)
def train_and_evaluate_models(data):
    try:
        # Use only the feature columns for X
        X = data[st.session_state.feature_columns]
        y = data['Savings']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'metrics': {
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'R2': r2_score(y_test, y_pred)
                }
            }
        
        return results
    except Exception as e:
        print("❌ Error in model training:", traceback.format_exc())
        raise e

def get_financial_advice(user_input, financial_data):
    if not GEMINI_AVAILABLE:
        return "The Financial Advisor feature is currently unavailable. Please check your API key configuration."
    
    try:
        # Prepare context from financial data
        context = f"""
        Financial Summary:
        - Total Income: ${financial_data['total_income']:,.2f}
        - Total Expenses: ${financial_data['total_expenses']:,.2f}
        - Total Savings: ${financial_data['total_savings']:,.2f}
        - Average Savings Rate: {financial_data['avg_savings_rate']:.1f}%
        - Investment Return: {financial_data['investment_return']:.1f}%
        """
        
        # Create prompt
        prompt = f"""
        As a financial advisor, please provide advice based on the following financial data and user question.
        
        {context}
        
        User Question: {user_input}
        
        Please provide:
        1. A direct answer to the question
        2. Specific recommendations based on their financial data
        3. Actionable steps they can take
        """
        
        # Get response from Gemini
        try:
            response = model.generate_content(prompt)
            if response and hasattr(response, 'text'):
                return response.text
            else:
                return "I apologize, but I couldn't generate a proper response at this time. Please try again later."
        except Exception as api_error:
            st.error(f"API Error: {str(api_error)}")
            return "I encountered an error while trying to provide financial advice. Please try again with a different question or check the API configuration."
            
    except Exception as e:
        st.error(f"Error in get_financial_advice: {str(e)}")
        return "I apologize, but I'm having trouble processing your request. Please try again later."

def train_clustering_model(data):
    """Train a clustering model for financial profiles using actual data"""
    try:
        # Calculate financial ratios for better clustering
        data['Savings_Rate'] = (data['Savings'] / data['Income']) * 100
        data['Expense_Ratio'] = (data['Expenses'] / data['Income']) * 100
        data['Investment_Ratio'] = (data['Savings'] / data['Expenses']) * 100
        
        # Prepare features for clustering
        features = ['Savings_Rate', 'Expense_Ratio', 'Investment_Ratio']
        X = data[features].copy()
        
        # Handle any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Define cluster centers based on typical financial profiles
        initial_centers = np.array([
            [20, 60, 33],    # Conservative Saver: 20% savings, 60% expenses, 33% investment ratio
            [30, 50, 60],    # Balanced Investor: 30% savings, 50% expenses, 60% investment ratio
            [40, 40, 100],   # Growth Seeker: 40% savings, 40% expenses, 100% investment ratio
            [10, 80, 12],    # Needs Improvement: 10% savings, 80% expenses, 12% investment ratio
            [25, 55, 45]     # Financial Builder: 25% savings, 55% expenses, 45% investment ratio
        ])
        
        # Scale the initial centers
        initial_centers_scaled = scaler.transform(initial_centers)
        
        # Train KMeans clustering with custom initialization
        kmeans = KMeans(
            n_clusters=5,
            init=initial_centers_scaled,
            n_init=1,
            random_state=42
        )
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster centers and characteristics
        cluster_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=features
        )
        
        # Analyze each cluster's characteristics
        cluster_characteristics = {}
        for i in range(5):
            cluster_data = data[clusters == i]
            cluster_characteristics[i] = {
                'avg_savings_rate': cluster_data['Savings_Rate'].mean(),
                'avg_expense_ratio': cluster_data['Expense_Ratio'].mean(),
                'avg_investment_ratio': cluster_data['Investment_Ratio'].mean(),
                'size': len(cluster_data)
            }
        
        # Define cluster names based on actual data characteristics
        cluster_names = {
            0: "Conservative Saver",
            1: "Balanced Investor",
            2: "Growth Seeker",
            3: "Needs Improvement",
            4: "Financial Builder"
        }
        
        return kmeans, scaler, cluster_names, features
    except Exception as e:
        st.error(f"Error in train_clustering_model: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None, None

def assign_cluster(profile_data, kmeans_model, scaler, cluster_names, features):
    """Assign a cluster to a user profile using the trained KMeans model"""
    try:
        # Calculate financial ratios for the profile
        income = float(profile_data['income'])
        expenses = float(profile_data['expenses'])
        savings = float(profile_data['savings'])
        
        # Calculate ratios with proper error handling
        savings_rate = (savings / income) * 100 if income > 0 else 0
        expense_ratio = (expenses / income) * 100 if income > 0 else 0
        investment_ratio = (savings / expenses) * 100 if expenses > 0 else 0
        
        # Display detailed debugging information
        st.write("### Financial Profile Analysis")
        st.write("#### Input Values:")
        st.write(f"- Income: ${income:,.2f}")
        st.write(f"- Expenses: ${expenses:,.2f}")
        st.write(f"- Savings: ${savings:,.2f}")
        
        st.write("#### Calculated Ratios:")
        st.write(f"- Savings Rate: {savings_rate:.2f}% (Savings/Income)")
        st.write(f"- Expense Ratio: {expense_ratio:.2f}% (Expenses/Income)")
        st.write(f"- Investment Ratio: {investment_ratio:.2f}% (Savings/Expenses)")
        
        # Create feature vector
        X = pd.DataFrame([[
            savings_rate,
            expense_ratio,
            investment_ratio
        ]], columns=features)
        
        # Handle any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Print raw feature vector
        st.write("### Raw Feature Vector")
        st.write(X)
        
        # Scale features
        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Print scaled feature vector
        st.write("### Scaled Feature Vector")
        st.write(X_scaled)
        
        # Calculate distances to all cluster centers
        distances = kmeans_model.transform(X_scaled)
        
        # Print distances to each cluster
        st.write("### Distances to Cluster Centers")
        for i, dist in enumerate(distances[0]):
            st.write(f"Distance to Cluster {i} ({cluster_names[i]}): {dist:.2f}")
        
        # Predict cluster
        cluster = kmeans_model.predict(X_scaled)[0]
        
        # Get cluster characteristics
        cluster_center = st.session_state.cluster_centers.iloc[cluster]
        cluster_stats = st.session_state.cluster_characteristics[cluster]
        
        st.write("#### Cluster Assignment Details:")
        st.write(f"Assigned Cluster: {cluster_names[cluster]}")
        st.write("#### Cluster Characteristics (Based on Training Data):")
        st.write(f"- Average Savings Rate: {cluster_stats['avg_savings_rate']:.2f}%")
        st.write(f"- Average Expense Ratio: {cluster_stats['avg_expense_ratio']:.2f}%")
        st.write(f"- Average Investment Ratio: {cluster_stats['avg_investment_ratio']:.2f}%")
        st.write(f"- Number of Similar Profiles: {cluster_stats['size']}")
        
        return cluster_names[cluster]
    except Exception as e:
        st.error(f"Error assigning cluster: {str(e)}")
        st.error("Full error details:")
        st.error(traceback.format_exc())
        return "Needs Improvement"

def create_navigation():
    # Create a centered dropdown menu using columns
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # Create the dropdown menu
        selected_page = st.selectbox(
            "Navigate to:",
            ["Dashboard", "Predictions", "Analysis", "Clustering Analysis", "Financial Advisor"],
            index=["Dashboard", "Predictions", "Analysis", "Clustering Analysis", "Financial Advisor"].index(st.session_state.current_page),
            label_visibility="collapsed",
            key="page_selector"
        )
        
        # Update the current page if selection changes
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()

def show_clustering_analysis():
    st.header("Financial Profile Clustering Analysis")
    
    if st.session_state.clustering_model is None:
        st.warning("Clustering model not initialized. Please wait for data processing to complete.")
        return
    
    # Get the data and model
    data = st.session_state.combined_data.copy()
    kmeans = st.session_state.clustering_model
    scaler = st.session_state.cluster_scaler
    features = st.session_state.cluster_features
    
    # Calculate ratios for visualization
    data['Savings_Rate'] = (data['Savings'] / data['Income']) * 100
    data['Expense_Ratio'] = (data['Expenses'] / data['Income']) * 100
    data['Investment_Ratio'] = (data['Savings'] / data['Expenses']) * 100
    
    # Prepare features for clustering
    X = data[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    # Scale the features
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Get cluster assignments
    clusters = kmeans.predict(X_scaled)
    data['Cluster'] = clusters
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Cluster Statistics", "Visualization", "Interactive Analysis"])
    
    with tab1:
        st.subheader("Clustering Overview")
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Profiles", len(data))
        with col2:
            st.metric("Number of Clusters", 5)
        with col3:
            st.metric("Average Cluster Size", len(data) // 5)
        
        # Display cluster distribution
        cluster_counts = data['Cluster'].value_counts()
        fig = px.pie(
            values=cluster_counts.values,
            names=[st.session_state.cluster_names[i] for i in cluster_counts.index],
            title='Cluster Distribution',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True, key="overview_pie")
        
        # Display cluster characteristics summary
        st.subheader("Cluster Characteristics Summary")
        cluster_stats = []
        for i in range(5):
            cluster_data = data[data['Cluster'] == i]
            stats = {
                'Cluster': st.session_state.cluster_names[i],
                'Size': len(cluster_data),
                'Avg_Savings_Rate': cluster_data['Savings_Rate'].mean(),
                'Avg_Expense_Ratio': cluster_data['Expense_Ratio'].mean(),
                'Avg_Investment_Ratio': cluster_data['Investment_Ratio'].mean()
            }
            cluster_stats.append(stats)
        
        stats_df = pd.DataFrame(cluster_stats)
        st.dataframe(stats_df.style.format({
            'Avg_Savings_Rate': '{:.2f}%',
            'Avg_Expense_Ratio': '{:.2f}%',
            'Avg_Investment_Ratio': '{:.2f}%'
        }))
    
    with tab2:
        st.subheader("Detailed Cluster Statistics")
        
        # Calculate detailed cluster statistics
        cluster_stats = []
        for i in range(5):
            cluster_data = data[data['Cluster'] == i]
            stats = {
                'Cluster': st.session_state.cluster_names[i],
                'Size': len(cluster_data),
                'Avg_Savings_Rate': cluster_data['Savings_Rate'].mean(),
                'Avg_Expense_Ratio': cluster_data['Expense_Ratio'].mean(),
                'Avg_Investment_Ratio': cluster_data['Investment_Ratio'].mean(),
                'Avg_Income': cluster_data['Income'].mean(),
                'Avg_Expenses': cluster_data['Expenses'].mean(),
                'Avg_Savings': cluster_data['Savings'].mean(),
                'Min_Savings_Rate': cluster_data['Savings_Rate'].min(),
                'Max_Savings_Rate': cluster_data['Savings_Rate'].max(),
                'Min_Expense_Ratio': cluster_data['Expense_Ratio'].min(),
                'Max_Expense_Ratio': cluster_data['Expense_Ratio'].max()
            }
            cluster_stats.append(stats)
        
        # Display detailed statistics
        stats_df = pd.DataFrame(cluster_stats)
        st.dataframe(stats_df.style.format({
            'Avg_Savings_Rate': '{:.2f}%',
            'Avg_Expense_Ratio': '{:.2f}%',
            'Avg_Investment_Ratio': '{:.2f}%',
            'Avg_Income': '${:,.2f}',
            'Avg_Expenses': '${:,.2f}',
            'Avg_Savings': '${:,.2f}',
            'Min_Savings_Rate': '{:.2f}%',
            'Max_Savings_Rate': '{:.2f}%',
            'Min_Expense_Ratio': '{:.2f}%',
            'Max_Expense_Ratio': '{:.2f}%'
        }))
        
        # Display range plots for each cluster
        st.subheader("Cluster Ranges")
        for i in range(5):
            cluster_data = data[data['Cluster'] == i]
            fig = go.Figure()
            
            # Add box plots for each ratio
            for ratio in ['Savings_Rate', 'Expense_Ratio', 'Investment_Ratio']:
                fig.add_trace(go.Box(
                    y=cluster_data[ratio],
                    name=ratio.replace('_', ' '),
                    boxpoints='all'
                ))
            
            fig.update_layout(
                title=f'Distribution of Financial Ratios for {st.session_state.cluster_names[i]}',
                yaxis_title='Percentage (%)',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True, key=f"cluster_range_{i}")
    
    with tab3:
        st.subheader("Cluster Visualization")
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            data,
            x='Savings_Rate',
            y='Expense_Ratio',
            z='Investment_Ratio',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Set1,
            title='3D Cluster Visualization',
            labels={
                'Savings_Rate': 'Savings Rate (%)',
                'Expense_Ratio': 'Expense Ratio (%)',
                'Investment_Ratio': 'Investment Ratio (%)'
            }
        )
        st.plotly_chart(fig, use_container_width=True, key="3d_scatter")
        
        # Create 2D scatter plots
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.scatter(
                data,
                x='Savings_Rate',
                y='Expense_Ratio',
                color='Cluster',
                color_discrete_sequence=px.colors.qualitative.Set1,
                title='Savings Rate vs Expense Ratio',
                labels={
                    'Savings_Rate': 'Savings Rate (%)',
                    'Expense_Ratio': 'Expense Ratio (%)'
                }
            )
            st.plotly_chart(fig1, use_container_width=True, key="2d_scatter_1")
        
        with col2:
            fig2 = px.scatter(
                data,
                x='Savings_Rate',
                y='Investment_Ratio',
                color='Cluster',
                color_discrete_sequence=px.colors.qualitative.Set1,
                title='Savings Rate vs Investment Ratio',
                labels={
                    'Savings_Rate': 'Savings Rate (%)',
                    'Investment_Ratio': 'Investment Ratio (%)'
                }
            )
            st.plotly_chart(fig2, use_container_width=True, key="2d_scatter_2")
    
    with tab4:
        st.subheader("Interactive Cluster Analysis")
        
        # Add interactive filters
        col1, col2 = st.columns(2)
        with col1:
            selected_cluster = st.selectbox(
                "Select Cluster to Analyze",
                options=list(st.session_state.cluster_names.values())
            )
        
        # Get cluster index
        cluster_idx = list(st.session_state.cluster_names.values()).index(selected_cluster)
        
        # Filter data for selected cluster
        cluster_data = data[data['Cluster'] == cluster_idx]
        
        # Display cluster characteristics
        st.write(f"### Characteristics of {selected_cluster}")
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Average Savings Rate",
                f"{cluster_data['Savings_Rate'].mean():.1f}%"
            )
        with col2:
            st.metric(
                "Average Expense Ratio",
                f"{cluster_data['Expense_Ratio'].mean():.1f}%"
            )
        with col3:
            st.metric(
                "Average Investment Ratio",
                f"{cluster_data['Investment_Ratio'].mean():.1f}%"
            )
        
        # Create box plots for selected cluster
        fig = go.Figure()
        for feature in ['Savings_Rate', 'Expense_Ratio', 'Investment_Ratio']:
            fig.add_trace(go.Box(
                y=cluster_data[feature],
                name=feature.replace('_', ' '),
                boxpoints='all'
            ))
        
        fig.update_layout(
            title=f'Distribution of Financial Ratios for {selected_cluster}',
            yaxis_title='Percentage (%)',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, key="interactive_box")
        
        # Display sample profiles
        st.write("### Sample Profiles in this Cluster")
        sample_profiles = cluster_data.sample(min(5, len(cluster_data)))
        st.dataframe(sample_profiles[[
            'Income', 'Expenses', 'Savings',
            'Savings_Rate', 'Expense_Ratio', 'Investment_Ratio'
        ]].style.format({
            'Income': '${:,.2f}',
            'Expenses': '${:,.2f}',
            'Savings': '${:,.2f}',
            'Savings_Rate': '{:.1f}%',
            'Expense_Ratio': '{:.1f}%',
            'Investment_Ratio': '{:.1f}%'
        }))
        
        # Add cluster comparison
        st.write("### Cluster Comparison")
        comparison_data = []
        for i in range(5):
            cluster_data = data[data['Cluster'] == i]
            comparison_data.append({
                'Cluster': st.session_state.cluster_names[i],
                'Savings_Rate': cluster_data['Savings_Rate'].mean(),
                'Expense_Ratio': cluster_data['Expense_Ratio'].mean(),
                'Investment_Ratio': cluster_data['Investment_Ratio'].mean()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        fig = go.Figure()
        for ratio in ['Savings_Rate', 'Expense_Ratio', 'Investment_Ratio']:
            fig.add_trace(go.Bar(
                name=ratio.replace('_', ' '),
                x=comparison_df['Cluster'],
                y=comparison_df[ratio],
                text=[f'{v:.1f}%' for v in comparison_df[ratio]],
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Comparison of Financial Ratios Across Clusters',
            barmode='group',
            yaxis_title='Percentage (%)',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True, key="cluster_comparison")

def main():
    try:
        # Load custom CSS
        load_css()
        
        # Initialize session state for navigation if not exists
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Dashboard'
        
        # Show welcome message only if it hasn't been shown before
        if 'welcome_shown' not in st.session_state:
            show_welcome_message()
            st.session_state.welcome_shown = True
        
        # Show centered title
        show_centered_title()
        
        # Create navigation
        create_navigation()
        
        # Get current page from session state
        current_page = st.session_state.current_page
        
        # Initialize session state for data
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
            st.session_state.budget = None
            st.session_state.transactions = None
            st.session_state.monthly_summary = None
            st.session_state.investment = None
            st.session_state.combined_data = None
            st.session_state.scaler = None
            st.session_state.le = None
            st.session_state.results = None
            
            # Initialize clustering model and related components
            st.session_state.clustering_model = None
            st.session_state.cluster_scaler = None
            st.session_state.cluster_names = {
                0: "Conservative Saver",
                1: "Balanced Investor",
                2: "Growth Seeker",
                3: "Needs Improvement",
                4: "Financial Builder"
            }
            st.session_state.cluster_features = ['Savings_Rate', 'Expense_Ratio', 'Investment_Ratio']
            st.session_state.cluster_centers = None
            st.session_state.cluster_characteristics = None

        # Load data if not already loaded
        if not st.session_state.data_loaded:
            with st.spinner("Loading data..."):
                budget, transactions, monthly_summary, investment = load_datasets()
                
                if budget is None:
                    st.error("Failed to load data. Please check the error messages above.")
                    return
                
                st.session_state.budget = budget
                st.session_state.transactions = transactions
                st.session_state.monthly_summary = monthly_summary
                st.session_state.investment = investment
                
                with st.spinner("Processing data..."):
                    budget, transactions, monthly_summary, investment = engineer_features(
                        budget, transactions, monthly_summary, investment
                    )
                    combined_data, scaler, le = preprocess_data(
                        budget, transactions, monthly_summary, investment
                    )
                    results = train_and_evaluate_models(combined_data)
                    
                    # Initialize clustering model
                    kmeans, cluster_scaler, cluster_names, features = train_clustering_model(combined_data)
                    if kmeans is not None:
                        st.session_state.clustering_model = kmeans
                        st.session_state.cluster_scaler = cluster_scaler
                        st.session_state.cluster_names = cluster_names
                        st.session_state.cluster_features = features
                        
                        # Calculate and store cluster centers and characteristics
                        X = combined_data[features].copy()
                        X = X.replace([np.inf, -np.inf], np.nan)
                        X = X.fillna(0)
                        X_scaled = cluster_scaler.transform(X)
                        clusters = kmeans.predict(X_scaled)
                        
                        # Store cluster centers
                        st.session_state.cluster_centers = pd.DataFrame(
                            cluster_scaler.inverse_transform(kmeans.cluster_centers_),
                            columns=features
                        )
                        
                        # Calculate and store cluster characteristics
                        cluster_characteristics = {}
                        for i in range(5):
                            cluster_data = combined_data[clusters == i]
                            cluster_characteristics[i] = {
                                'avg_savings_rate': cluster_data['Savings_Rate'].mean(),
                                'avg_expense_ratio': cluster_data['Expense_Ratio'].mean(),
                                'avg_investment_ratio': cluster_data['Investment_Ratio'].mean(),
                                'size': len(cluster_data)
                            }
                        st.session_state.cluster_characteristics = cluster_characteristics
                    
                    st.session_state.combined_data = combined_data
                    st.session_state.scaler = scaler
                    st.session_state.le = le
                    st.session_state.results = results
                    st.session_state.data_loaded = True

        # Calculate financial metrics for the chatbot
        financial_data = {
            'total_income': st.session_state.monthly_summary['Income'].sum(),
            'total_expenses': st.session_state.monthly_summary['Expenses'].sum(),
            'total_savings': st.session_state.monthly_summary['Savings'].sum(),
            'avg_savings_rate': (st.session_state.monthly_summary['Savings'].sum() / st.session_state.monthly_summary['Income'].sum()) * 100,
            'investment_return': ((st.session_state.investment['Current_Value'].sum() - st.session_state.investment['Amount_Invested'].sum()) / st.session_state.investment['Amount_Invested'].sum()) * 100
        }

        # Display the appropriate page content
        if current_page == "Clustering Analysis":
            show_clustering_analysis()
        elif current_page == "Financial Advisor":
            st.header("AI Financial Advisor")
            
            # Initialize session state for user profile if not exists
            if 'user_profile' not in st.session_state:
                st.session_state.user_profile = {
                    'age': None,
                    'income': None,
                    'expenses': None,
                    'savings': None,
                    'investment_amount': None,
                    'risk_tolerance': None,
                    'financial_goals': None,
                    'cluster': None
                }
            
            # User Profile Input Section
            with st.expander("Enter Your Financial Profile", expanded=True):
                with st.form("user_profile_form"):
                    st.subheader("Personal Information")
                    age = st.number_input("Age", min_value=18, max_value=100, value=25)
                    income = st.number_input("Monthly Income ($)", min_value=0, value=5000)
                    expenses = st.number_input("Monthly Expenses ($)", min_value=0, value=3000)
                    savings = st.number_input("Current Savings ($)", min_value=0, value=10000)
                    investment_amount = st.number_input("Current Investment Amount ($)", min_value=0, value=5000)
                    
                    risk_tolerance = st.select_slider(
                        "Risk Tolerance",
                        options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"],
                        value="Moderate"
                    )
                    
                    # Modified Financial Goals section
                    st.write("### Financial Goals")
                    default_goals = ["Retirement Planning", "Buying a House", "Education Fund", "Emergency Fund", "Wealth Building", "Debt Repayment"]
                    selected_goals = st.multiselect(
                        "Select from common goals",
                        default_goals,
                        default=["Emergency Fund"]
                    )
                    
                    # Add custom goal input
                    custom_goal = st.text_input("Add a custom financial goal (optional)")
                    if custom_goal:
                        selected_goals.append(custom_goal)
                    
                    submit_profile = st.form_submit_button("Save Profile")
            
            if submit_profile:
                # Update user profile
                profile_data = {
                    'age': age,
                    'income': income,
                    'expenses': expenses,
                    'savings': savings,
                    'investment_amount': investment_amount,
                    'risk_tolerance': risk_tolerance,
                    'financial_goals': selected_goals
                }
                
                # Only assign cluster if we have a trained model
                if st.session_state.clustering_model is not None:
                    # Clear previous cluster assignment
                    st.session_state.user_profile['cluster'] = None
                    
                    # Get new cluster assignment
                    cluster = assign_cluster(
                        profile_data,
                        st.session_state.clustering_model,
                        st.session_state.cluster_scaler,
                        st.session_state.cluster_names,
                        st.session_state.cluster_features
                    )
                    profile_data['cluster'] = cluster
                else:
                    profile_data['cluster'] = "Needs Improvement"
                
                # Update session state
                st.session_state.user_profile = profile_data
                
                # Show success message with animation
                st.markdown("""
                <div class="success-message">
                    Profile saved successfully! 🎉
                </div>
                """, unsafe_allow_html=True)
            
            # Display User Profile Summary with enhanced UI
            if st.session_state.user_profile['age'] is not None:
                st.markdown("""
                <div class="stCard">
                    <h2 style="color: var(--primary-color);">Your Financial Profile</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="stCard">
                        <h3 style="color: var(--primary-color);">Personal Details</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(f"👤 Age: {st.session_state.user_profile['age']}")
                    st.write(f"🎯 Risk Tolerance: {st.session_state.user_profile['risk_tolerance']}")
                    st.write(f"📊 Financial Cluster: {st.session_state.user_profile['cluster']}")
                
                with col2:
                    st.markdown("""
                    <div class="stCard">
                        <h3 style="color: var(--primary-color);">Financial Metrics</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(f"💰 Monthly Income: ${st.session_state.user_profile['income']:,.2f}")
                    st.write(f"💸 Monthly Expenses: ${st.session_state.user_profile['expenses']:,.2f}")
                    st.write(f"💎 Current Savings: ${st.session_state.user_profile['savings']:,.2f}")
                    st.write(f"📈 Investment Amount: ${st.session_state.user_profile['investment_amount']:,.2f}")
                
                st.markdown("""
                <div class="stCard">
                    <h3 style="color: var(--primary-color);">Financial Goals</h3>
                </div>
                """, unsafe_allow_html=True)
                for goal in st.session_state.user_profile['financial_goals']:
                    st.write(f"🎯 {goal}")
            
            # Chat interface
            st.subheader("Ask for Financial Advice")
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("What would you like to know about your finances?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response with enhanced context
                with st.chat_message("assistant"):
                    # Prepare enhanced context with user profile
                    if st.session_state.user_profile['age'] is not None:
                        enhanced_context = f"""
                        User Profile:
                        - Age: {st.session_state.user_profile['age']}
                        - Monthly Income: ${st.session_state.user_profile['income']:,.2f}
                        - Monthly Expenses: ${st.session_state.user_profile['expenses']:,.2f}
                        - Current Savings: ${st.session_state.user_profile['savings']:,.2f}
                        - Investment Amount: ${st.session_state.user_profile['investment_amount']:,.2f}
                        - Risk Tolerance: {st.session_state.user_profile['risk_tolerance']}
                        - Financial Goals: {', '.join(st.session_state.user_profile['financial_goals'])}
                        - Financial Cluster: {st.session_state.user_profile['cluster']}
                        
                        Financial Summary:
                        - Total Income: ${financial_data['total_income']:,.2f}
                        - Total Expenses: ${financial_data['total_expenses']:,.2f}
                        - Total Savings: ${financial_data['total_savings']:,.2f}
                        - Average Savings Rate: {financial_data['avg_savings_rate']:.1f}%
                        - Investment Return: {financial_data['investment_return']:.1f}%
                        """
                    else:
                        enhanced_context = f"""
                        Financial Summary:
                        - Total Income: ${financial_data['total_income']:,.2f}
                        - Total Expenses: ${financial_data['total_expenses']:,.2f}
                        - Total Savings: ${financial_data['total_savings']:,.2f}
                        - Average Savings Rate: {financial_data['avg_savings_rate']:.1f}%
                        - Investment Return: {financial_data['investment_return']:.1f}%
                        """
                    
                    # Create enhanced prompt
                    enhanced_prompt = f"""
                    As a financial advisor, please provide personalized advice based on the following user profile, financial data, and user question.
                    
                    {enhanced_context}
                    
                    User Question: {prompt}
                    
                    Please provide:
                    1. A direct answer to the question
                    2. Specific recommendations based on their profile and financial data
                    3. Actionable steps they can take
                    4. How their current financial cluster affects their situation
                    """
                    
                    response = get_financial_advice(enhanced_prompt, financial_data)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        elif current_page == "Dashboard":
            st.header("Financial Dashboard")

            # Convert monetary columns to numerical values
            for col in ['Income', 'Expenses', 'Savings']:
                if st.session_state.monthly_summary[col].dtype == 'object':
                    st.session_state.monthly_summary[col] = st.session_state.monthly_summary[col].str.replace('$', '').str.replace(',', '').astype(float)

            # Key Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Income", f"${st.session_state.monthly_summary['Income'].sum():,.2f}")
            with col2:
                st.metric("Total Expenses", f"${st.session_state.monthly_summary['Expenses'].sum():,.2f}")
            with col3:
                st.metric("Total Savings", f"${st.session_state.monthly_summary['Savings'].sum():,.2f}")

            # Interactive Monthly Trends
            st.subheader("Monthly Trends")
            
            # Add month names for better readability
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            # Convert Month to integer and handle any NaN values
            st.session_state.monthly_summary['Month'] = pd.to_numeric(st.session_state.monthly_summary['Month'], errors='coerce').fillna(1).astype(int)
            st.session_state.monthly_summary['Month_Name'] = st.session_state.monthly_summary['Month'].apply(lambda x: month_names[x-1] if 1 <= x <= 12 else 'Unknown')

            # Metric selection dropdown
            selected_metric = st.selectbox(
                "Select Metric to View",
                ["Income", "Expenses", "Savings"],
                key="metric_selector"
            )
            
            # Create interactive plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.monthly_summary['Month_Name'],
                y=st.session_state.monthly_summary[selected_metric],
                mode='lines+markers',
                name=selected_metric,
                line=dict(width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f"Monthly {selected_metric} Trend",
                xaxis_title="Month",
                yaxis_title=f"{selected_metric} ($)",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary statistics
            st.subheader("Summary Statistics")
            stats_df = pd.DataFrame({
                'Metric': [selected_metric],
                'Mean': [st.session_state.monthly_summary[selected_metric].mean()],
                'Median': [st.session_state.monthly_summary[selected_metric].median()],
                'Min': [st.session_state.monthly_summary[selected_metric].min()],
                'Max': [st.session_state.monthly_summary[selected_metric].max()]
            })
            st.dataframe(stats_df.style.format({
                'Mean': '${:,.2f}',
                'Median': '${:,.2f}',
                'Min': '${:,.2f}',
                'Max': '${:,.2f}'
            }))

        elif current_page == "Predictions":
            st.header("Savings Predictions")
            
            # Model Performance Section
            st.subheader("Model Performance")
            model_perf = pd.DataFrame({
                'Model': [],
                'MAE ($)': [],
                'RMSE ($)': [],
                'R² Score': []
            })
            
            for model_name, result in st.session_state.results.items():
                metrics = result['metrics']
                model_perf = pd.concat([model_perf, pd.DataFrame({
                    'Model': [model_name],
                    'MAE ($)': [metrics['MAE']],
                    'RMSE ($)': [metrics['RMSE']],
                    'R² Score': [metrics['R2']]
                })], ignore_index=True)
            
            # Display model performance metrics with better formatting
            st.dataframe(model_perf.style.format({
                'MAE ($)': '${:,.2f}',
                'RMSE ($)': '${:,.2f}',
                'R² Score': '{:.3f}'
            }))
            
            # Create a separate container for user input
            with st.expander("Enter Your Financial Data", expanded=True):
                with st.form("prediction_form"):
                    st.subheader("Financial Information")
                    
                    # Income Section
                    st.write("### Income Details")
                    income = st.number_input("Monthly Income", min_value=0.0, step=100.0, value=5000.0)
                    
                    # Expenses Section
                    st.write("### Expense Details")
                    expenses = st.number_input("Monthly Expenses", min_value=0.0, step=100.0, value=3000.0)
                    
                    # Budget Section
                    st.write("### Budget Details")
                    budgeted = st.number_input("Budgeted Amount", min_value=0.0, step=100.0, value=4000.0)
                    category = st.selectbox("Category", st.session_state.le.classes_)
                    
                    submit = st.form_submit_button("Predict Savings")

            if submit:
                try:
                    # Process input data
                    category_encoded = st.session_state.le.transform([category])[0]
                    
                    # Create input DataFrame with only the feature columns
                    input_df = pd.DataFrame({
                        'Month': [1],
                        'Income': [income],
                        'Expenses': [expenses],
                        'Budgeted': [budgeted],
                        'Category': [category_encoded]
                    })
                    
                    # Ensure columns are in the same order as during training
                    input_df = input_df[st.session_state.feature_columns]
                    
                    # Scale only the numerical columns
                    numerical_cols = ['Income', 'Expenses', 'Budgeted']
                    input_df[numerical_cols] = st.session_state.scaler.transform(input_df[numerical_cols])
                    
                    # Make predictions
                    predictions = {}
                    for model_name, result in st.session_state.results.items():
                        pred = result['model'].predict(input_df)[0]
                        predictions[model_name] = pred
                    
                    # Display predictions
                    st.subheader("Predicted Savings")
                    pred_df = pd.DataFrame({
                        'Model': list(predictions.keys()),
                        'Predicted Savings': list(predictions.values())
                    })
                    
                    # Create a bar chart of predictions
                    fig = go.Figure(data=[
                        go.Bar(
                            x=pred_df['Model'],
                            y=pred_df['Predicted Savings'],
                            text=[f'${v:,.2f}' for v in pred_df['Predicted Savings']],
                            textposition='auto',
                        )
                    ])
                    
                    fig.update_layout(
                        title="Predicted Savings by Model",
                        xaxis_title="Model",
                        yaxis_title="Predicted Savings ($)",
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed predictions
                    st.dataframe(pred_df.style.format({
                        'Predicted Savings': '${:,.2f}'
                    }))
                    
                    # Add insights
                    st.subheader("Insights")
                    avg_prediction = np.mean(list(predictions.values()))
                    st.write(f"Average predicted savings: ${avg_prediction:,.2f}")
                    st.write(f"Based on your income of ${income:,.2f} and expenses of ${expenses:,.2f}, ")
                    if avg_prediction > 0:
                        st.success(f"You are predicted to save ${avg_prediction:,.2f} this month!")
                    else:
                        st.warning(f"You are predicted to have a deficit of ${abs(avg_prediction):,.2f} this month.")
                        
                except Exception as e:
                    st.error(f"Error making predictions: {str(e)}")
                    st.error("Please check your input values and try again.")
                    st.error(f"Debug info - Feature columns: {st.session_state.feature_columns}")
                    st.error(f"Debug info - Input DataFrame columns: {input_df.columns.tolist()}")

        elif current_page == "Analysis":
            st.header("Financial Analysis")
            
            # Analysis Type Selection
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Expense Analysis", "Income Analysis", "Savings Analysis", "Investment Analysis"]
            )
            
            if analysis_type == "Expense Analysis":
                st.subheader("Category-wise Expenses")
                
                # Create interactive bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=st.session_state.budget['Category'],
                        y=st.session_state.budget['Budget'],
                        text=[f'${v:,.2f}' for v in st.session_state.budget['Budget']],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Budget Allocation by Category",
                    xaxis_title="Category",
                    yaxis_title="Budget Amount ($)",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add expense distribution pie chart
                fig2 = go.Figure(data=[
                    go.Pie(
                        labels=st.session_state.budget['Category'],
                        values=st.session_state.budget['Budget'],
                        hole=.3
                    )
                ])
                
                fig2.update_layout(
                    title="Expense Distribution",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Add expense insights
                st.subheader("Expense Insights")
                total_budget = st.session_state.budget['Budget'].sum()
                st.write(f"Total Budget: ${total_budget:,.2f}")
                st.write(f"Number of Categories: {len(st.session_state.budget)}")
                st.write(f"Average Budget per Category: ${total_budget/len(st.session_state.budget):,.2f}")
                
            elif analysis_type == "Income Analysis":
                st.subheader("Income Analysis")
                
                # Monthly income trend
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.monthly_summary['Month_Name'],
                    y=st.session_state.monthly_summary['Income'],
                    mode='lines+markers',
                    name='Income',
                    line=dict(width=3)
                ))
                
                fig.update_layout(
                    title="Monthly Income Trend",
                    xaxis_title="Month",
                    yaxis_title="Income ($)",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Income statistics
                st.subheader("Income Statistics")
                income_stats = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Min', 'Max', 'Standard Deviation'],
                    'Value': [
                        st.session_state.monthly_summary['Income'].mean(),
                        st.session_state.monthly_summary['Income'].median(),
                        st.session_state.monthly_summary['Income'].min(),
                        st.session_state.monthly_summary['Income'].max(),
                        st.session_state.monthly_summary['Income'].std()
                    ]
                })
                
                st.dataframe(income_stats.style.format({
                    'Value': '${:,.2f}'
                }))
                
            elif analysis_type == "Savings Analysis":
                st.subheader("Savings Analysis")
                
                # Calculate savings rate
                st.session_state.monthly_summary['Savings_Rate'] = (st.session_state.monthly_summary['Savings'] / st.session_state.monthly_summary['Income']) * 100
                
                # Savings trend
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.monthly_summary['Month_Name'],
                    y=st.session_state.monthly_summary['Savings'],
                    mode='lines+markers',
                    name='Savings',
                    line=dict(width=3)
                ))
                
                fig.update_layout(
                    title="Monthly Savings Trend",
                    xaxis_title="Month",
                    yaxis_title="Savings ($)",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Savings rate trend
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=st.session_state.monthly_summary['Month_Name'],
                    y=st.session_state.monthly_summary['Savings_Rate'],
                    mode='lines+markers',
                    name='Savings Rate',
                    line=dict(width=3)
                ))
                
                fig2.update_layout(
                    title="Monthly Savings Rate Trend",
                    xaxis_title="Month",
                    yaxis_title="Savings Rate (%)",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Savings insights
                st.subheader("Savings Insights")
                avg_savings_rate = st.session_state.monthly_summary['Savings_Rate'].mean()
                st.write(f"Average Savings Rate: {avg_savings_rate:.1f}%")
                if avg_savings_rate >= 20:
                    st.success("Great job! Your savings rate is above the recommended 20%!")
                else:
                    st.warning("Consider increasing your savings rate to at least 20% for better financial security.")
                
            elif analysis_type == "Investment Analysis":
                st.subheader("Investment Analysis")
                
                # Investment performance
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.investment['Date'],
                    y=st.session_state.investment['Current_Value'],
                    mode='lines+markers',
                    name='Current Value',
                    line=dict(width=3)
                ))
                
                fig.update_layout(
                    title="Investment Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Investment returns
                st.session_state.investment['Return'] = ((st.session_state.investment['Current_Value'] - st.session_state.investment['Amount_Invested']) / st.session_state.investment['Amount_Invested']) * 100
                
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=st.session_state.investment['Date'],
                    y=st.session_state.investment['Return'],
                    text=[f'{v:.1f}%' for v in st.session_state.investment['Return']],
                    textposition='auto',
                ))
                
                fig2.update_layout(
                    title="Investment Returns by Date",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Investment insights
                st.subheader("Investment Insights")
                total_invested = st.session_state.investment['Amount_Invested'].sum()
                current_value = st.session_state.investment['Current_Value'].sum()
                total_return = ((current_value - total_invested) / total_invested) * 100
                
                st.write(f"Total Amount Invested: ${total_invested:,.2f}")
                st.write(f"Current Portfolio Value: ${current_value:,.2f}")
                st.write(f"Total Return: {total_return:.1f}%")
                
                if total_return > 0:
                    st.success(f"Your investments are performing well with a {total_return:.1f}% return!")
                else:
                    st.warning("Your investments are currently underperforming. Consider reviewing your investment strategy.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Full traceback:")
        st.code(traceback.format_exc())
        print("Error:", str(e))
        print("Traceback:", traceback.format_exc())

if __name__ == "__main__":
    main()
