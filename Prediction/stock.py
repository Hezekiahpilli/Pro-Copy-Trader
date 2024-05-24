import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from yahoo_fin import news
import sqlite3
import bcrypt

# Database connection
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
''')
conn.commit()

# Function to add a new user
def add_user(username, password):
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
    conn.commit()

# Function to authenticate user
def authenticate_user(username, password):
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    data = c.fetchone()
    if data and bcrypt.checkpw(password.encode('utf-8'), data[0]):
        return True
    return False

# Add custom CSS for styling, including a background image
def add_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background: url("https://img.freepik.com/free-vector/gradient-connection-background_23-2150516350.jpg?w=740&t=st=1716546280~exp=1716546880~hmac=7d2c2085ad035ac39189d836cebf35a8109e259cf17e63df5cd7669620c447be") no-repeat center center fixed; 
            background-size: cover;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }
        .stButton>button {
            background-color: #ff6347;
            color: white;
            border-radius: 10px;
            font-size: 16px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #ff4500;
        }
        .stTextInput>div>div>input {
            background-color: #fffacd;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
        .stSlider>div {
            background: linear-gradient(to right, #ff6347, #ffa07a);
            border-radius: 10px;
        }
        .stSidebar {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 10px;
        }
        .stSelectbox>div {
            border-radius: 5px;
        }
        .login-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            max-width: 400px;
            margin: auto;
        }
        .register-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            max-width: 400px;
            margin: auto.
        }
        .register-link, .login-link {
            color: #ff6347;
            text-decoration: none;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Function to get the data
def get_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    return df

# Function to preprocess data
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Timestamp'] = df['Date'].map(pd.Timestamp.timestamp)
    X = df[['Timestamp']].values
    y = df['Close'].values
    return X, y

# Function to train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return model, rmse

# Function to get news articles
def get_news(ticker):
    return news.get_yf_rss(ticker)

# Main app function
def main_app():
    add_custom_css()  # Apply custom CSS

    st.title("Cryptocurrency Price Prediction")

    # Select cryptocurrency
    st.sidebar.header("User Input")
    ticker = st.sidebar.selectbox("Select cryptocurrency", ["BTC-USD", "ETH-USD", "LTC-USD"])

    # Get and preprocess data
    st.header("Loading and Preprocessing Data")
    data_load_state = st.text("Loading data...")
    df = get_data(ticker)
    X, y = preprocess_data(df)
    data_load_state.text("Loading data... done!")

    # Display raw data
    st.subheader("Raw Data")
    st.write(df)

    # Train model
    st.header("Model Training")
    model, rmse = train_model(X, y)
    st.write(f"Model trained with RMSE: {rmse:.2f}")

    # Predict future prices
    st.header("Predict Future Prices")
    last_date = df['Date'].max()
    end_date = pd.Timestamp('2025-01-01')
    days_to_predict = (end_date - last_date).days
    future_dates = pd.date_range(start=last_date, periods=days_to_predict + 1).to_series().map(pd.Timestamp.timestamp).values.reshape(-1, 1)
    future_pred = model.predict(future_dates)

    # Display predictions
    future_dates = pd.date_range(start=last_date, periods=days_to_predict + 1)[1:]
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_pred[1:]})
    st.subheader("Predicted Prices")
    st.write(pred_df)

    # Plot predictions
    st.subheader("Prediction Chart")
    st.line_chart(pred_df.set_index("Date"))

    # Plot historical data and predictions
    st.subheader("Historical and Predicted Prices")
    combined_df = pd.concat([df.set_index("Date")["Close"], pred_df.set_index("Date")["Predicted Close"]])
    st.line_chart(combined_df)

    # Fetch and display news
    st.header("Latest News")
    news_articles = get_news(ticker)
    for article in news_articles:
        st.subheader(article['title'])
        st.write(article['link'])

# Registration page function
def registration_page():
    add_custom_css()  # Apply custom CSS
    st.title("Register")
    with st.form(key='register_form'):
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        submit_button = st.form_submit_button(label='Register')
    if submit_button:
        if username and password:
            add_user(username, password)
            st.success("User registered successfully! Please login.")
            st.session_state['show_register'] = False
        else:
            st.error("Please provide a username and password")

# Login page function
def login_page():
    add_custom_css()  # Apply custom CSS
    st.title("Login")
    with st.form(key='login_form'):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label='Login')
    if submit_button:
        if authenticate_user(username, password):
            st.session_state['authenticated'] = True
        else:
            st.error("Invalid username or password")

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'show_register' not in st.session_state:
    st.session_state['show_register'] = False

# Show login or registration page if not authenticated
if not st.session_state['authenticated']:
    if st.session_state['show_register']:
        registration_page()
    else:
        login_page()
else:
    main_app()

# Option to show registration page
if not st.session_state['authenticated'] and not st.session_state['show_register']:
    if st.button('Register'):
        st.session_state['show_register'] = True
        st.experimental_rerun()

# Option to go back to login page from registration
if not st.session_state['authenticated'] and st.session_state['show_register']:
    if st.button('Back to Login'):
        st.session_state['show_register'] = False
        st.experimental_rerun()