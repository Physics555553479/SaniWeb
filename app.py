import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from datetime import date


if 'portfolio_confirmed' not in st.session_state:
    st.session_state.portfolio_confirmed = False # Safety feature to ensure that if steady state does not exist, it's set to false


# Main header
st.markdown("<h1 style='color:#FFD700; margin-top: -40px;'>Portfolio Optimisation Project by Sanit</h1>", unsafe_allow_html=True)


# Cache the data download to speed up repeated calls with same tickers
@st.cache_data
def download_data(tickers): # Downloading data function
    if len(tickers) == 0:
        return pd.DataFrame()  # Return empty if no tickers chosen
    return yf.download(tickers, start='2018-01-01', end=date.today().strftime('%Y-%m-%d'))

# Load tickers from Excel
df = pd.read_excel("Companies.xlsx", header=None)
tickers = df[0].tolist()
view_tickers = df[1].tolist()

st.warning("""
⚠️ **Disclaimer:** This application is provided as-is for educational and informational purposes only.
The portfolio results are generated using historical data and are optimized to maximize the Sharpe ratio mathematically, without real-world diversification or regulatory constraints.
This is not financial advice.
While efforts have been made to ensure accuracy, occasional bugs or unexpected behavior may occur.
Please refresh or reload the page if needed.
Thank you for using the app!
""")

st.info("⚠️ Loading results for more than 10 companies AND 20000 portfolios may slow down the program.")

if 'num_sims' not in st.session_state:
    st.session_state.num_sims = 10000 # Default value of slider of 10000 when not in session

num_sims = st.slider("Number of Portfolios to Simulate", 1000, 30000, st.session_state.num_sims)
st.session_state.num_sims = num_sims # Equating statments equal. 

if st.checkbox("Show list of Companies: "):
    for ticker, name in zip(tickers, view_tickers):  # Visible tickers and names
        st.write(f"{ticker} - {name}")

chosen_tickers = st.multiselect(  # Adding widget
    "Select tickers for your portfolio:",
    options=tickers,
    format_func=lambda x: f"{x} - {view_tickers[tickers.index(x)]}"
)
st.session_state.chosen_tickers = chosen_tickers

@st.cache_data
def calculate_growthdf(data, tickers): # More efficient than a for loop. 
    growth = 100 * data['Close'].pct_change().dropna()
    return growth.reset_index()

if st.button("Confirm Portfolio:"):
    if len(chosen_tickers) == 0:
        st.error("Please select at least one ticker")
    else:
        st.session_state.portfolio_confirmed = True # Ensures that steady state is set to true, when 'confirm portfolio' clicked
        st.success(f"Your portfolio is {chosen_tickers}")

        # Download data (cached)
        data = download_data(chosen_tickers)
        st.session_state.data = data # Defining steady state like dictionary
        growthdf = calculate_growthdf(data, chosen_tickers)
        st.session_state.growthdf = growthdf

        cov_matrix = growthdf.drop(columns=['Date']).cov() # Covariance matrix
        weights = np.random.rand(num_sims, len(growthdf.drop(columns=['Date']).columns))  
        weights /= weights.sum(axis=1, keepdims=True)
        expected_return = weights @ growthdf.drop(columns=['Date']).mean() # Dot product of weighs and average returns (25000x1)
        port_var = np.zeros(len(weights)) # For loop required for variances to be a scalar instead of 2D
        one_dimensional_matrix = weights @ cov_matrix
        port_var = np.einsum('ij,ij->i', one_dimensional_matrix, weights)
        port_vol = np.sqrt(port_var)    
        base_rate = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1] # Risk free rate (govt.bonds)
        risk_free_rate = (1 + base_rate/100)**(1/252) - 1
        sharpe_ratio = (expected_return - risk_free_rate)/port_vol # Sharpe Ratio

        # Store sample results in session state
        st.session_state.cov_matrix = cov_matrix
        st.session_state.weights = weights
        st.session_state.expected_return = expected_return
        st.session_state.port_vol = port_vol
        st.session_state.risk_free_rate = risk_free_rate
        st.session_state.sharpe_ratio = sharpe_ratio

        # Formatting sample percentages per company. 
        sample_optimal_percentages = weights[sharpe_ratio.argmax()] * 100
        formatted_sample_percentages = [f"{ticker} - {perc:.2f}%" for ticker, perc in zip(chosen_tickers, sample_optimal_percentages)]
        sample_best_sharpe_ratio = (weights[sharpe_ratio.argmax()] @ growthdf.drop(columns=['Date']).mean() - risk_free_rate) / (np.sqrt(weights[sharpe_ratio.argmax()] @ (cov_matrix @ weights[sharpe_ratio.argmax()])))
        st.session_state.sample_best_sharpe_ratio = sample_best_sharpe_ratio
        st.session_state.sample_optimal_percentages = sample_optimal_percentages
        st.session_state.formatted_sample_percentages = formatted_sample_percentages

        
        # Finding real optimal proportions
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(len(growthdf.drop(columns=['Date']).mean()))) 
        guess = np.ones(len(growthdf.drop(columns=['Date']).mean())) / len(growthdf.drop(columns=['Date']).mean())
        result = minimize(lambda w: - (np.dot(w, growthdf.drop(columns=['Date']).mean()) - risk_free_rate) / np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
            guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        # Store true results in session state
        optimal_sharpe_ratio = -result.fun
        optimal_percentages = (result.x * 100)
        formatted_percentages = [f"{ticker} - {x:.2f}%" for ticker ,x in zip(chosen_tickers, optimal_percentages)]
        st.session_state.optimal_sharpe_ratio = optimal_sharpe_ratio
        st.session_state.optimal_percentages = optimal_percentages 
        st.session_state.formatted_percentages = formatted_percentages
        
        # Plotting Efficient Frontier
        min_sharpe = 0
        max_sharpe = sample_best_sharpe_ratio
        fig, ax = plt.subplots()
        scatter = ax.scatter(port_vol, expected_return, c=sharpe_ratio, cmap='viridis',
                            s=10, alpha=0.7, vmin=0, vmax=max_sharpe)
        fig.colorbar(scatter, ax =ax, label='Sharpe Ratio')
        ax.scatter(np.sqrt(np.dot(weights[sharpe_ratio.argmax()], np.dot(cov_matrix, weights[sharpe_ratio.argmax()]))), 
                    np.dot(weights[sharpe_ratio.argmax()], growthdf.drop(columns=['Date']).mean()),
                    color = 'red', marker = 'o', s = 10, alpha = 0.9, label = 'Sample Best Sharpe Ratio') # Labelling Sample Best
        ax.scatter(np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x))), 
                    np.dot(result.x, growthdf.drop(columns=['Date']).mean()),
                    color = 'blue', marker = 'o', s = 10, alpha = 0.9, label = 'Best Possible Sharpe Ratio') # Labelling Optimal Best
        ax.set_xlabel('Portfolio Volatility (%)')
        ax.set_ylabel('Expected Return (%)')
        ax.set_title('Efficient Frontier - Portfolio Simulations')
        ax.legend()
        st.session_state.fig = fig


if st.session_state.portfolio_confirmed == True:

    if st.checkbox("Show Sample Results"):
        st.write("Based on historic data, the sample optimal percentages:")
        st.write("[", ", ".join(st.session_state.formatted_sample_percentages), "]")
        st.write(f"Sample optimal Sharpe ratio: {st.session_state.sample_best_sharpe_ratio:.4f}")

    if st.checkbox("Show True Optimal Results"):
        st.write("Based on historic data, the true optimal portfolio allocation:")
        st.write("[", ", ".join(st.session_state.formatted_percentages), "]")
        st.write(f"True optimal Sharpe ratio: {st.session_state.optimal_sharpe_ratio:.4f}")
        st.markdown("<br>", unsafe_allow_html=True)  # Add vertical space
    
    if st.checkbox("Show Efficient Frontier Graph"):
        st.pyplot(st.session_state.fig)
    
    st.markdown("<br>", unsafe_allow_html=True)  # More vertical space
    
    # Show closing prices in their own column (full width)
    for ticker in st.session_state.chosen_tickers:
        show_prices = st.checkbox(f"Show {ticker} Closing Prices", key=ticker)
        if show_prices:
            prices = st.session_state.data['Close'][ticker]
            st.write(f"**{ticker} Closing Prices:**")
            st.line_chart(prices)
            st.markdown("<br>", unsafe_allow_html=True)  # Space between charts


