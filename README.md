# Personal Finance Manager

An interactive web application for managing personal finances, analyzing spending patterns, and getting AI-powered financial advice.

## Features

- 📊 Interactive Dashboard with real-time financial metrics
- 📈 Detailed financial analysis and visualizations
- 🤖 AI-powered financial advisor using Gemini API
- 💰 Savings predictions using machine learning models
- 📱 User-friendly interface with Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Personal_Finance_Manager.git
cd Personal_Finance_Manager
```

2. Create and activate a virtual environment:
```bash
python -m venv myvenv
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up your Gemini API key:
   - Create a `.streamlit/secrets.toml` file
   - Add your API key: `GOOGLE_API_KEY = "your-api-key-here"`

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Use the sidebar to navigate between different sections:
   - Dashboard: View key financial metrics and trends
   - Predictions: Get savings predictions based on your data
   - Analysis: Explore detailed financial analysis
   - Financial Advisor: Get AI-powered financial advice

## Data Structure

The application uses the following datasets:
- Budget data
- Transaction history
- Monthly summaries
- Investment portfolio

All datasets should be placed in the `Preprocessed Datasets` directory.

## Contributing

Feel free to submit issues and enhancement requests!
