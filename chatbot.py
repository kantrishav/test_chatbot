import yfinance as yf
from openai import OpenAI
import streamlit as st
import openai
import warnings
st.set_page_config(layout="wide")

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
<style>
#Main Menu {visibility: hidden; }
footer {visibility: hidden; }
header {visibility: hidden;}
</style>
"""
st.markdown (hide_st_style, unsafe_allow_html=True)


#--------------------------------------
warnings.filterwarnings('ignore')


st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stApp {
        background-color:       #000000; 
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <h1 style='color: #FFFFFF;'>FinOpt.ai</h1>
    """, unsafe_allow_html=True
)

#st.write("""# FinStock.ai : An AI Option & Futures Analyzer 💹""")





#Dow Jones 
ticker = '^DJI'
ticker_sp500 = '^GSPC'
dow_data = yf.download(ticker, period='5d' ,  progress=False)
qqq_data  = yf.download('QQQ', period='5d' ,  progress=False)
sp500_data1  = yf.download(ticker_sp500, period='5d' , progress=False)

gold_data  = yf.download('GC=F', period='5d' , progress=False)
gold_data = gold_data.sort_index(ascending=False)

oil_data  = yf.download('CL=F', period='5d' , progress=False)
oil_data = oil_data.sort_index(ascending=False)




sp500_kpi = round(float(sp500_data1.iloc[4]["Close"]), 2)




prev_day_sp500 = round(float(sp500_data1.iloc[3]["Close"]), 2)
today_sp500 = round(float(sp500_data1.iloc[4]["Close"]), 2)
change_sp500 = ((today_sp500-prev_day_sp500)/prev_day_sp500)*100
change_sp500 = round(float(change_sp500),2)
change_sp500 = f"{change_sp500} %"



dow_kpi = round(float(dow_data.iloc[4]["Close"]), 2)
dow_prev  = round(float(dow_data.iloc[3]["Close"]), 2)
dow_change =  ((dow_kpi-dow_prev)/dow_prev)*100
dow_change = round(float(dow_change),2)
dow_change = f"{dow_change} %"



qqq_kpi = round(float(qqq_data.iloc[4]["Close"]), 2)
qqq_prev  = round(float(qqq_data.iloc[3]["Close"]), 2)
qqq_change =  ((qqq_kpi-qqq_prev)/dow_prev)*100
qqq_change = round(float(qqq_change),2)
qqq_change = f"{qqq_change} %"


gold_kpi = round(float(gold_data.iloc[0]["Close"]), 2)
gold_prev  = round(float(gold_data.iloc[1]["Close"]), 2)
gold_change =  ((gold_kpi-gold_prev)/gold_prev)*100
gold_change = round(float(gold_change),2)
gold_change = f"{gold_change} %"

oil_kpi = round(float(oil_data.iloc[0]["Close"]), 2)
oil_prev  = round(float(oil_data.iloc[1]["Close"]), 2)
oil_change =  ((oil_kpi-oil_prev)/oil_prev)*100
oil_change = round(float(oil_change),2)
oil_change = f"{oil_change} %"



var = '#000000'



col2_html = f"""
    <style>
    .kpi-box {{
        background-color: {var};
        padding: 10px;
        border-radius: 10px;
        display: flex;
        flex-direction: column; /* Stack items vertically */
        justify-content: center;
        align-items: center;
        height: 120px;
        width: 230px;
        font-size: 1.2em;
        font-weight: bold;
        color: #0A64EE;
    }}
    .kpi-label {{
        font-size: 24px; 
        font-weight: bold; 
        color: #0ff550;
        margin-bottom: 1px; /* Space below the label */
    }}
    .kpi-value {{
        font-size: 32px;
        margin-bottom: 1px; /* Space below the value */
    }}
        .kpi-delta {{
        font-size: 20px;
        margin-bottom: 1px; /* Space below the value */
    }}
    </style>

    <div class='kpi-box'>
        <div class='kpi-label'>Dow Jones</div>
        <div class='kpi-value'>{dow_kpi}</div>
        <div class='kpi-delta'>{dow_change}</div>
    </div>
"""



#---------------------- Column Test ----------


# Create the first row with 3 columns
col1, col2, col3 , col4 , col5 = st.columns(5)

# Add content to each column in the first row
with col1:
    
        st.markdown(
        f"""
        <div class='kpi-box'>
        <div style="color: #0ff550; font-size: 24px; font-weight: bold;">SPY 500</div>
            <div class='kpi-value'>{sp500_kpi}</div>
            <div class='kpi-delta'>{change_sp500}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    


with col2:
    #st.markdown("<h2 style='text-align: center;'>🎈</h2>", unsafe_allow_html=True)
    st.write(col2_html, unsafe_allow_html=True)


# Render the custom-styled version (if needed)




with col3:
        st.markdown(
        f"""
        <div class='kpi-box'>
        <div style="color: #0ff550; font-size: 24px; font-weight: bold;">QQQ</div>
            <div class='kpi-value'>{qqq_kpi}</div>
            <div class='kpi-delta'>{qqq_change}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col4:
        st.markdown(
        f"""
        <div class='kpi-box'>
        <div style="color: #0ff550; font-size: 24px; font-weight: bold;">Gold</div>
            <div class='kpi-value'>{gold_kpi}</div>
            <div class='kpi-delta'>{gold_change}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )


with col5:
        st.markdown(
        f"""
        <div class='kpi-box'>
        <div style="color: #0ff550; font-size: 24px; font-weight: bold;">Crude Oil</div>
            <div class='kpi-value'>{oil_kpi}</div>
            <div class='kpi-delta'>{oil_change}</div>
        </div>
        """, 
        unsafe_allow_html=True
    )




#--------------------------------





#----------------------------------


st.write("   ")
st.write("   ")
st.write("   ")
st.write("   ")

client = OpenAI(
    # This is the default and can be omitted
    api_key= "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",
)

st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #FFFFFF;'> Do You Want To Analyze Stock Options Using AI ? </h1>
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #8c8c8c; font-size: 20px;'>Choose the best option strategy for you.Your personal option analyzer !</h1>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #0ff550; font-size: 20px;'>Your Personal Intelligent Options Trade Advisor! Ask Anything ! Ex- What are the top iron condors combinations for TSLA next week expiry?</h1>
    </div>
    """,
    unsafe_allow_html=True
)



# st.markdown(
#     """
#     <div style='text-align: center;'>
#         <h1 style='color: #ffffff; font-size: 20px;'>Ask Anything ! Ex- What are the top iron condors combinations for TSLA next week expiry? </h1>
#     </div>
#     """,
#     unsafe_allow_html=True
# )



st.markdown(
    """
    <style>
    /* Center the text area by setting its container's max-width and using margin auto */
    div[data-testid="stTextArea"] {
        max-width: 950px; /* Adjust as needed to control the width */
        margin: 0 auto; /* Centers the container horizontally */
        padding: 10px; /* Optional padding for additional space */
        position: relative; /* Enable positioning for child elements */
    }
    div[data-testid="stTextArea"] textarea {
        background-color: #f0f0f5; /* Light gray background */
        color: #333333; /* Dark text color */
        border: 2px solid #3498db; /* Blue border */
        border-radius: 10px;
        padding: 10px;
        width: 100%; /* Ensure the textarea fills its container */
    }
    .icon-container {
        position: absolute; /* Position the icons relative to the text area */
        top: -310px; /* Adjust this value to position icons above the textarea */
        left: 50%; /* Center the icons horizontally */
        transform: translateX(-50%); /* Center alignment adjustment */      
    }
    .icon-container img {
        width: 100px; /* Adjust icon size */
        height: 45px; /* Adjust icon size */
        margin-left: 30px; /* Space between icons */
    }
    </style>
    """,
    unsafe_allow_html=True
)



#option_text_input = st.text_area('', height=310)

#option_prompt = "You are an AI Option trader. Below is the text format of Option Chain of an asset. Analyze and share top 3 strategies using option chain data uploaded and share how can we build that strategies with the data shared.  Here is the option chain text : "

#option_final_prompt = option_prompt + option_text_input





# # Add the icons in the bottom right corner
# st.markdown(
#     """
#     <div class="icon-container">
#         <img src="https://thevyatergroup.com/wp-content/uploads/2021/03/logo-amazon-404px-grey.png" alt="Icon 1">
#         <img src="https://www.krenerbookkeeping.com/wp-content/uploads/2018/07/logo-microsoft-404px-grey.png" alt="Icon 2">
#         <img src="https://mohamadfaizal.com/wp-content/uploads/2017/05/logo-google-404px-grey.png" alt="Icon 3">

        
#     </div>
#     """,
#     unsafe_allow_html=True
# )






# if len(option_final_prompt) > 500:
#     chat_completion = client.chat.completions.create( messages=[{"role": "user","content": option_final_prompt,}],model="gpt-3.5-turbo",)
#     op = chat_completion.choices[0].message.content
#     op = op.replace('\n', '<br>')
#     st.markdown(f'<p style="color:white;">{op}</p>', unsafe_allow_html=True)
#     #st.markdown(f'<p style="color:white;">{op}</p>', unsafe_allow_html=True)
#     #st.write(op)
# else:
#     st.write('')
 
#----------------------------------------------------FinChat---------------------------------------------

from phi.agent import Agent
#, Tool
from phi.tools import Toolkit
import requests
import pandas as pd

    
from phi.model.openai.chat import OpenAIChat
#from secret_key import openapi_key  # Assuming 'secret_key.py' contains your OpenAI API key
import os
#os.environ['OPENAI_API_KEY'] = openapi_key

os.environ['OPENAI_API_KEY'] = st.secrets["openai"]["api_key"]



from phi.tools import Toolkit
import requests
import pandas as pd

class OptionsTool(Toolkit):
    def __init__(self):
        super().__init__(name="options_tool")
        self.register(self.get_options_data)

    def get_options_data(self, symbol: str) -> str:
        """
        Fetch options data for a given stock symbol.
        
        Args:
            symbol (str): The stock ticker symbol (e.g., 'AAPL', 'QQQ').
        
        Returns:
            str: Options data as a string (JSON format).
        """
        try:
            # Dynamically create the URL using the symbol provided
            url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            options_data = data['data']['options']
            options_data = options_data[:3]  # Limit to 5 options for brevity
            df = pd.json_normalize(options_data)
            return df.to_json(orient="index")
        except Exception as e:
            return f"Error fetching options data for {symbol}: {e}"

# Example to test the agent with a dynamic query
from phi.agent import Agent
from phi.tools import Toolkit

# Create an agent instance
agent = Agent(
    tools=[OptionsTool()],
    model=OpenAIChat(id="gpt-4"),
    show_tool_calls=True,
    description="You are a financial assistant that fetches options data for different stock symbols.",
    instructions=["When a stock ticker is mentioned in a query, fetch the options data for that ticker."]
)

# Sample query that includes a dynamic ticker

query = "Fetch the options data for MICROSOFT"


#query = "Fetch the options data for MICROSOFT and give the top 3 iron condor strategy"


# Phi's internal system automatically handles ticker extraction
#agent.print_response(query)

#response = agent.print_response(query, stream=False)

#import json
#import io
#import sys
#import re



#u_query = st.text_input("Drop Your Query")
#response = agent.run(u_query,stream=False)


from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.agents import AgentExecutor
from langchain.agents import AgentExecutor, create_react_agent, load_tools



#query = st.text_input("Ask about options data for a stock:") 
query = st.text_area('', height=250)
query = (query + " Show the output in strctured format using tables")


if len(query)>50:
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        response = agent.run(query)
        content = response.content
        st.write(content)


# #-----------------------------------------------------------------------------

#agent.print_response("Fetch the options data for MICROSOFT and give the top 3 iron condor strategy", stream=False)

st.write("Rishav Kant Gen AI Development")




#--------------------- test-----------------------------------


#-------------------- TICKER ANALYSIS  START ---------------------------------#


import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm


st.markdown(""" <h1 style='color: #FFFFFF;'>Ticker Analysis</h1> """, unsafe_allow_html=True)

B1,B2,B3= st.columns(3)

# Add content to each column in the first row
with B1:
    quote = st.selectbox("Select Ticker",('SPY', 'AAPL','TSLA' , 'MSFT' , 'EPI' , 'SMH', 'RTH', '_NDX', '_RUT', 'DBA', 'XHB', 'ARKG', 'ARKF',
              'EWW', 'VNQ', 'HYG', 'XLP', 'XLU' ,'^SPX', 'XOP' ,'LQD', 'ARKK', 'XLF', 'SLV', 'EEM',
              'HYG', 'IWM', 'QQQ', 'FXI', 'XLE', 'KWEB', 'TLT', 'EWZ', 'EFA', 'GDX', 'DIA', 'GLD'),key = 'sq1',)


with B2:
    yf_ticker = yf.Ticker(quote)
    expiration_dates = yf_ticker.options
    selec_date = st.selectbox("Select the Expiration Date",(expiration_dates),key = 'sq2',)   


with B3:
    period_selected = st.selectbox("Observation Period for % Price Change",(1,7,14 ,21,30,90,180),key = 'ps1',) 



def op_chain(quote):
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{quote}.json"
    response = requests.get(url)
    data = response.json()
    options_data = data['data']['options']
    df = pd.json_normalize(options_data)
    df['expiry'] = df['option'].str[-15:-9]
    df['type']  = df['option'].str[-9:-8]
    df['strike'] = df['option'].str[-8:]
    df['left'] = df['strike'].str[:5]     # Get the first 5 characters (significant digits)
    df['right'] = df['strike'].str[-3:]   # Get the last 3 characters (fractional part)
    df['strike'] = (df['left'] + df['right']).astype(float) / 1000 
    df['expiry'] = pd.to_datetime(df['expiry'], format='%y%m%d')
    op_chain_df = df[df['expiry'] == selec_date]
    temp_df = op_chain_df
    call = temp_df[temp_df['type'] == 'C']
    put = temp_df[temp_df['type'] == 'P']
    op_chain_df = pd.merge(call, put, on="strike" , suffixes=("_call", "_put"))
    op_chain_df.columns= op_chain_df.columns.str.upper()
    return op_chain_df

op_chain_df = op_chain(quote)




#STANDARD DEV


apple_stock = yf_ticker
data = apple_stock.history(period="5y")
# Calculate daily change (percentage change)
data['Daily Change'] = data['Close'].pct_change(periods= period_selected) * 100  # Percentage change

# Drop the first row as it will be NaN after pct_change
data = data.dropna()

# Calculate the mean and standard deviation of the daily change
mean = data['Daily Change'].mean()
std_dev = data['Daily Change'].std()

# Calculate 1st, 2nd, and 3rd standard deviations
first_sd = mean + std_dev
second_sd = mean + 2 * std_dev
third_sd = mean + 3 * std_dev

# Plotting using Plotly
fig = go.Figure()

# Add the histogram
fig.add_trace(go.Histogram(
    x=data['Daily Change'],
    nbinsx=50,
    histnorm='probability density',
    name='Daily Change',
    opacity=0.75,
    marker=dict(color='#0ff550')
))

# Add the normal distribution fit
x = np.linspace(data['Daily Change'].min(), data['Daily Change'].max(), 100)
p = norm.pdf(x, mean, std_dev)
fig.add_trace(go.Scatter(
    x=x, y=p,
    mode='lines',
    name=f'Normal fit: μ={mean:.2f}, σ={std_dev:.2f}',
    line=dict(color='blue',width = 4)  # Normal line color set to yellow
))

# Add lines for 1st, 2nd, and 3rd standard deviations on both sides (positive and negative)
fig.add_trace(go.Scatter(
    x=[first_sd, first_sd],
    y=[0, max(p)],
    mode='lines',
    name="1st SD",
    line=dict(color='blue', dash='dash')
))

fig.add_trace(go.Scatter(
    x=[-first_sd, -first_sd],
    y=[0, max(p)],
    mode='lines',
    name="1st SD",
    line=dict(color='blue', dash='dash')
))

fig.add_trace(go.Scatter(
    x=[second_sd, second_sd],
    y=[0, max(p)],
    mode='lines',
    name="2nd SD",
    line=dict(color='orange', dash='dash')
))

fig.add_trace(go.Scatter(
    x=[-second_sd, -second_sd],
    y=[0, max(p)],
    mode='lines',
    name="2nd SD",
    line=dict(color='orange', dash='dash')
))

fig.add_trace(go.Scatter(
    x=[third_sd, third_sd],
    y=[0, max(p)],
    mode='lines',
    name="3rd SD",
    line=dict(color='red', dash='dash')
))

fig.add_trace(go.Scatter(
    x=[-third_sd, -third_sd],
    y=[0, max(p)],
    mode='lines',
    name="3rd SD",
    line=dict(color='red', dash='dash')
))

# Add annotations for SD values
fig.add_annotation(
    x=first_sd,
    y=max(p) * 0.8,
    text=f"1st SD: {first_sd:.2f}%",
    showarrow=True,
    arrowhead=2,
    ax=20,
    ay=-30,
    font=dict(size=12, color='blue')
)

fig.add_annotation(
    x=-first_sd,
    y=max(p) * 0.8,
    text=f"1st SD: {-first_sd:.2f}%",
    showarrow=True,
    arrowhead=2,
    ax=20,
    ay=-30,
    font=dict(size=12, color='blue')
)

fig.add_annotation(
    x=second_sd,
    y=max(p) * 0.7,
    text=f"2nd SD: {second_sd:.2f}%",
    showarrow=True,
    arrowhead=2,
    ax=20,
    ay=-30,
    font=dict(size=12, color='orange')
)

fig.add_annotation(
    x=-second_sd,
    y=max(p) * 0.7,
    text=f"2nd SD: {-second_sd:.2f}%",
    showarrow=True,
    arrowhead=2,
    ax=20,
    ay=-30,
    font=dict(size=12, color='orange')
)

fig.add_annotation(
    x=third_sd,
    y=max(p) * 0.6,
    text=f"3rd SD: {third_sd:.2f}%",
    showarrow=True,
    arrowhead=2,
    ax=20,
    ay=-30,
    font=dict(size=12, color='red')
)

fig.add_annotation(
    x=-third_sd,
    y=max(p) * 0.6,
    text=f"3rd SD: {-third_sd:.2f}%",
    showarrow=True,
    arrowhead=2,
    ax=20,
    ay=-30,
    font=dict(size=12, color='red')
)

# Update layout
fig.update_layout(
    title='',
    xaxis_title='Daily Percentage Change (%)',
    yaxis_title='Density',
    showlegend=False,
    plot_bgcolor='black',  # Set the background color of the plot area
    paper_bgcolor='black',  # Set the background color of the entire figure
    font=dict(color='white'),  # Set font color to white for contrast
    yaxis=dict(showgrid=False) 
)


#STOCK TREND 



stock_data =yf_ticker
data = stock_data.history(period='5y')  # Fetch last 5 years of data

# Calculate 100-day and 200-day moving averages
data['MA100'] = data['Close'].rolling(window=100).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

ma_100 = data['MA100'].iloc[-1]
ma_200 = data['MA200'].iloc[-1]
stock_price_today = data['Close'].iloc[-1]
stock_price_today = f"{stock_price_today:.2f}"

if ma_100 > ma_200:
    ma_signal = 'Buy'
else:
    ma_signal = 'Sell'


if ma_signal == 'Buy':
    signal_color = '#0ff550'
else:
     signal_color = '#f2073e'


# Create the line chart for stock closing prices and moving averages
fig_trend = go.Figure()

# Plot the stock closing prices
fig_trend.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name=f'{quote} Stock Price',
    line=dict(color='yellow', width=2)
))

# Plot the 100-day moving average
fig_trend.add_trace(go.Scatter(
    x=data.index,
    y=data['MA100'],
    mode='lines',
    name='100-Day MA',
    line=dict(color='green', width=2, dash='dash')  # Green dashed line for 100-Day MA
))

# Plot the 200-day moving average
fig_trend.add_trace(go.Scatter(
    x=data.index,
    y=data['MA200'],
    mode='lines',
    name='200-Day MA',
    line=dict(color='red', width=2, dash='dash')  # Red dashed line for 200-Day MA
))

# Update layout for better visuals and set background to black
fig_trend.update_layout(
#    title=f'{quote} Stock Trend (Last 5 Years) with Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price (USD)',
    xaxis_rangeslider_visible=False,  # Remove range slider
    plot_bgcolor='black',  # Set the plot area background color to black
    paper_bgcolor='black',  # Set the overall background color to black
    font=dict(color='white'),  # Set the font color to white for contrast
    template='plotly_dark' , # Use dark theme
    yaxis=dict(showgrid=False)
)





#-------------



# Custom CSS for tabs
st.markdown("""
<style>

        .stTabs [data-baseweb="tab-list"] {
                gap: 15px;
    }

        .stTabs [data-baseweb="tab"] {
                height: 0px;
        width: 200px;
        white-space: pre-wrap;
                background-color: #FFFFFF;
                border-radius: 15px 15px 15px 15px;
                gap: 80px;
        color: black;
                padding-top: 18px;
                padding-bottom: 18px;
        
  
    }

        .stTabs [aria-selected="true"] {
                background-color: #EC3637;
    border-bottom: 5px solid black;
        }

</style>""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Option Chain", "Stock Analysis", "Trend Chart"])

# Tab content
with tab1:
    st.dataframe(op_chain_df)

with tab2:
    st.plotly_chart(fig,use_container_width=True)


with tab3:
    st.markdown(f""" <div style='text-align: right;'> <h1 style='color:{signal_color}; font-size: 20px;'>{quote} : {stock_price_today} |     Trade Signal : {ma_signal} </h1> </div> """, unsafe_allow_html=True)
    st.plotly_chart(fig_trend,use_container_width=True, key = 't3')






#-------------TICKER ANALYSIS END --------------------#









#---------Betting App------------------------------------------------------


curve_kpi = f"""
    <style>
    .kpi-box-curve {{
        background-color: {var};
        padding: 10px;
        border-radius: 10px;
        display: flex;
        flex-direction: column; /* Stack items vertically */
        justify-content: center;
        align-items: center;
        height: 120px;
        width: 500px;
        font-size: 1.2em;
        font-weight: bold;
        color: #0A64EE;
    }}
    .kpi-label-curve {{
        font-size: 24px; 
        font-weight: bold; 
        color: #0ff550;
        margin-bottom: 1px; /* Space below the label */
    }}
    .kpi-value-curve {{
        font-size: 32px;
        margin-bottom: 1px; /* Space below the value */
    }}
    </style>
"""




st.markdown(
    """
    <h1 style='color: #FFFFFF;'>Option Probablity Simulator</h1>
    """, unsafe_allow_html=True
)


import random

def simulate_betting(initial_portfolio, risk_reward_ratio, win_probability,
                     frequency_of_trade, num_bets_at_once, bet_percentage):
  """
  Simulates a betting strategy over a year.

  Args:
    initial_portfolio: Starting amount of money.
    risk_reward_ratio: Ratio of potential loss to potential win (e.g., 1:1).
    win_probability: Probability of winning a single bet (between 0 and 1).
    frequency_of_trade: Number of days between bets.
    num_bets_at_once: Number of bets placed simultaneously.
    bet_percentage: Percentage of the portfolio to use for each bet.

  Returns:
    A tuple containing:
      - final_portfolio_value: Portfolio value at the end of the year.
      - max_drawdown: Maximum percentage decline from a peak value.
      - wins: Number of winning bets.
      - losses: Number of losing bets.
  """

  portfolio_value = initial_portfolio
  max_portfolio_value = initial_portfolio
  max_drawdown = 0
  wins = 0
  losses = 0

  for day in range(0, 365, frequency_of_trade):
    bet_amount = portfolio_value * (bet_percentage / 100)

    for _ in range(num_bets_at_once):
      if random.random() < win_probability:
        portfolio_value += bet_amount
        wins += 1
      else:
        portfolio_value -= bet_amount
        losses += 1

    max_portfolio_value = max(max_portfolio_value, portfolio_value)
    drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
    max_drawdown = max(max_drawdown, drawdown)

  return portfolio_value, max_drawdown, wins, losses

st.markdown(
    """
    <style>
    /* Targeting all labels within the Streamlit app */
    .stNumberInput > label ,.stSlider > label ,.stSelectbox > label ,.stTextInput > label ,.stMultiSelect  > label  {
        color: #00FF00;  /* Set label color to green */
        font-weight: bold; /* Optionally, make the text bold */
    }

      /* Customize the slider track and thumb */
    .stSlider > div > div > div > input[type=range] {
        accent-color: #00FF00;  /* Change the slider's accent color (supported in most modern browsers) */
    }


    </style>
    """,
    unsafe_allow_html=True
)

# Create the first row with 3 columns
c1, c2, c3 , c4 , c5 , c6 , c7 ,c8= st.columns(8)

# Add content to each column in the first row
with c1:
    initial_portfolio = st.number_input("Initial Investment", value = 100000)

with c2:
    risk_reward_ratio = st.number_input("Risk Reward Ratio", value=1)    # Not directly used in the calculation, but you have it as a parameter

with c3:
    win_probability = st.number_input("Win Probablity", value=0.7)

with c4:
    frequency_of_trade = st.number_input("Frequency Trade", value=14) 


with c5:
    num_bets_at_once = st.number_input("Number of Bets", value=1)

with c6:
    bet_percentage = st.number_input("Bet Percentage", value=2)

with c7:
    #sim_count = st.slider("No. Of Simulations", value=10)
    sim_count = st.selectbox(
    "No. Of Simulations",
    (10,50,100), index = 2,
)
    #sim_count = st.number_input("No. Of Simulations", value=10)

# Run a single simulation
final_portfolio_value, max_drawdown, wins, losses = simulate_betting(
    initial_portfolio, risk_reward_ratio, win_probability, frequency_of_trade,
    num_bets_at_once, bet_percentage
)




#st.write(final_portfolio_value)
#st.write(max_drawdown)

import pandas as pd
results = []
for i in range(sim_count):
  final_portfolio_value, max_drawdown, _, _ = simulate_betting(
      initial_portfolio, risk_reward_ratio, win_probability, frequency_of_trade,
      num_bets_at_once, bet_percentage
  )
  #st.write(f"Simulation {i+1}: Final Portfolio Value = {final_portfolio_value}, Max Drawdown = {max_drawdown*100:.2f}%")
  results.append({
        "Simulation": i + 1,
        "Final Portfolio Value": final_portfolio_value,
        "Max Drawdown (%)": max_drawdown * 100
    })

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

final_portfolio_value = int(final_portfolio_value)

st.markdown(
    f"""
    <div class='kpi-box' style="position: relative; left: 5px; top: 40px;">
        <div style="color: #0ff550; font-size: 30px; font-weight: bold;">Portfolio Value</div>
        <div class='kpi-value'>{final_portfolio_value}</div>
    </div>
    """, 
    unsafe_allow_html=True
)

draw_down_num = float(max_drawdown*100)
draw_down_num=f"{draw_down_num:.2f}%"

st.markdown(
    f"""
    <div class='kpi-box' style="position: relative; left: 300px; top: -75px;">
        <div style="color: #0ff550; font-size: 30px; font-weight: bold;">Max Drawdown</div>
        <div class='kpi-value'>{draw_down_num}</div>
    </div>
    """, 
    unsafe_allow_html=True
)

# You can now display or save the DataFrame
#st.write(df_results)

import plotly.express as px

fig_simulation = px.line(df_results, x="Simulation", y="Final Portfolio Value", title="",
              labels={"Simulation": "Simulation No", "Final Portfolio Value": "Final Portfolio Value"},
              markers=True)  # Add markers to the line

# Update layout for black background and no gridlines
fig_simulation.update_layout(
    paper_bgcolor='black',  # Background color of the chart
    plot_bgcolor='black',   # Background color of the plotting area
    font_color='white',      # Font color for text
    xaxis=dict(showgrid=False),  # Remove gridlines
    yaxis=dict(showgrid=False) ,  # Remove gridlines
    width=1400,  # Set width of the plot
    height=450
)

# Update line color and thickness
fig_simulation.update_traces(line=dict(color='#00FF00', width=4))
 
st.plotly_chart(fig_simulation)


#--------------------------------------------------------------------------



#----------- Term Structure -----------------------------------

st.markdown(
    """
    <h1 style='color: #FFFFFF;'>Option Volatility Term Structure</h1>
    """, unsafe_allow_html=True
)

T1, T2, T3 , T4 , T5 , T6 , T7 ,T8= st.columns(8)

# Add content to each column in the first row
with T1:
    ticker = st.selectbox("Select Ticker",('SPY', 'AAPL','TSLA' , 'MSFT' , 'EPI' , 'SMH', 'RTH', '_NDX', '_RUT', 'DBA', 'XHB', 'ARKG', 'ARKF',
              'EWW', 'VNQ', 'HYG', 'XLP', 'XLU' ,'^SPX', 'XOP' ,'LQD', 'ARKK', 'XLF', 'SLV', 'EEM',
              'HYG', 'IWM', 'QQQ', 'FXI', 'XLE', 'KWEB', 'TLT', 'EWZ', 'EFA', 'GDX', 'DIA', 'GLD'),)


with T2:
    yf_ticker = yf.Ticker(ticker)
    expiration_dates = yf_ticker.options
    selected_exp_date = st.selectbox("Select the Expiration Date",(expiration_dates),)   


options = yf_ticker.option_chain(selected_exp_date)
calls = options.calls
puts = options.puts


calls['Type'] = 'Call'
puts['Type']= 'Put'

puts['lastTradeDate'] = puts['lastTradeDate'].dt.date
puts['impliedVolatility'] = round(puts['impliedVolatility'],2)
puts = puts.rename(columns={'impliedVolatility': 'Imp Volatility'})
puts = puts.rename(columns={'strike': 'Strike Price'})


calls['lastTradeDate'] = calls['lastTradeDate'].dt.date
calls['impliedVolatility'] = round(calls['impliedVolatility'],2)
calls = calls.rename(columns={'impliedVolatility': 'Imp Volatility'})
calls = calls.rename(columns={'strike': 'Strike Price'})

comb_term_option = []
comb_term_option = pd.concat([calls, puts], ignore_index=True)
combo_fig = px.line(comb_term_option, x="Strike Price", y="Imp Volatility" , color = 'Type' , color_discrete_map={"Call": "#00FF00", "Put":"#0A64EE"} , markers = True)
combo_fig.update_layout(
    paper_bgcolor='black',  # Background color of the chart
    plot_bgcolor='black',   # Background color of the plotting area
    font_color='white',      # Font color for text
    xaxis=dict(showgrid=False , zeroline=False),  # Remove gridlines
    yaxis=dict(showgrid=False,zeroline=False) ,  # Remove gridlines
    width=1400,  # Set width of the plot
    height=450,
)

combo_fig.update_traces(line=dict(width=6))
st.plotly_chart(combo_fig)


#---------------------------------------------------------------------------------


#-------------Volatility Surface-------------------------------
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import streamlit as st

def plot_volatility_surface(ticker, num_expirations=10):
    # Fetch the stock object
    stock = yf.Ticker(ticker)

    # Get the available expiration dates
    expiration_dates = stock.options

    # Initialize lists to store data
    strike_prices = []
    expirations = []
    implied_vols = []

    # Loop through each expiration date and get option chain data
    for date in expiration_dates[:num_expirations]:  # Limit to the specified number of dates
        opt_chain = stock.option_chain(date)
        calls = opt_chain.calls
        puts = opt_chain.puts

        # Combine call and put data
        combined_data = pd.concat([calls[['strike', 'impliedVolatility']], puts[['strike', 'impliedVolatility']]])
        combined_data = combined_data.groupby('strike').mean().reset_index()

        # Filter out extreme or unrealistic implied volatilities
        combined_data = combined_data[(combined_data['impliedVolatility'] > 0) & (combined_data['impliedVolatility'] < 2)]

        # Store the data
        strike_prices.extend(combined_data['strike'].tolist())
        expirations.extend([date] * len(combined_data))
        implied_vols.extend(combined_data['impliedVolatility'].tolist())

    # Create a DataFrame
    data = pd.DataFrame({
        'Strike': strike_prices,
        'Expiration': expirations,
        'ImpliedVolatility': implied_vols
    })

    # Convert expiration dates to numeric values
    data['Expiration'] = pd.to_datetime(data['Expiration'])
    data['DaysToExpiration'] = (data['Expiration'] - data['Expiration'].min()).dt.days

    # Create a grid for interpolation
    strike_range = np.linspace(min(data['Strike']), max(data['Strike']), 100)
    days_range = np.linspace(min(data['DaysToExpiration']), max(data['DaysToExpiration']), 100)
    strike_grid, days_grid = np.meshgrid(strike_range, days_range)

    # Interpolate implied volatilities
    iv_values = griddata(
        (data['Strike'], data['DaysToExpiration']),
        data['ImpliedVolatility'],
        (strike_grid, days_grid),
        method='cubic'  # Use cubic interpolation for smoothness
    )

    # Plot the smoothed volatility surface using Plotly
    fig = go.Figure(data=[go.Surface(
        z=iv_values, 
        x=strike_range, 
        y=days_range, 
        colorscale=[[0, 'blue'], [1, '#00FF00']],  # Transition from blue to green
        cmin=0,
        cmax=np.nanmax(iv_values)
    )])

    # Customize the layout with a larger size and black background
    fig.update_layout(
        title=f'Smoothed Volatility Surface for {ticker} (Calls and Puts)',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiration',
            zaxis_title='Implied Volatility',
            bgcolor='black'  # Set the background color of the scene to black
        ),
        paper_bgcolor='#000000',  # Set the overall background color to black
        font=dict(color='white'),  # Set font color to white for better contrast
        width=1500,  # Set width of the plot
        height=1100   # Set height of the plot
    )

    return fig

# Streamlit App

st.markdown(
    """
    <h1 style='color: #FFFFFF;'>Option Volatility Surface Analyzer</h1>
    """, unsafe_allow_html=True
)

#st.title("Volatility Surface Viewer")

a,b,c,d,e,f,g = st.columns(7)

# Add content to each column in the first row
with a:
    ticker = st.text_input("Enter Ticker Symbol", "TSLA")
    


with b:
    st.markdown("""
    <style>
    /* Custom styling for all sliders */
    input[type="range"] {
        accent-color: #000000;  /* Desired color for the slider thumb and track */
    }
    </style>
""", unsafe_allow_html=True)
    num_expirations = st.slider("Number of Expiration Dates", 1, 20, 10)
    

# User Input for Ticker Symbol
#ticker = st.text_input("Enter Ticker Symbol", "TSLA")
#num_expirations = st.slider("Number of Expiration Dates to Fetch", 1, 20, 10)st.plotly_chart(fig)

# Plot the volatility surface
fig = plot_volatility_surface(ticker, num_expirations)
st.plotly_chart(fig)

#----------------------------------




#------------------------ Iron Condor Screener Section Start Added on 24th November  -----------------------#

import requests
st.markdown( f""" <h1 style='color: #FFFFFF;'>Option Strategy Scanner </h1> """, unsafe_allow_html=True)



P1, P2, P3 = st.columns(3)

# Add content to each column in the first row


with P1:
    strategy = st.selectbox("Pick Strategy",('Iron Condor','Bull Put Spread','Bear Call Spread'),)
    


with P2:
    ticker_ic = ['SPY', 'AAPL','TSLA' , 'MSFT' , 'EPI' , 'SMH', 'RTH', '_NDX', '_RUT', 'DBA', 'XHB', 'ARKG', 'ARKF',
              'EWW', 'VNQ', 'HYG', 'XLP', 'XLU' ,'^SPX', 'XOP' ,'LQD', 'ARKK', 'XLF', 'SLV', 'EEM',
              'HYG', 'IWM', 'QQQ', 'FXI', 'XLE', 'KWEB', 'TLT', 'EWZ', 'EFA', 'GDX', 'DIA', 'GLD']

    options = st.multiselect("Stocks/ETF",ticker_ic, max_selections = 5, default = 'QQQ')



with P3:
    dte = st.selectbox("Select DTE",(30,45,60,90,14),)


P4 , P5 , P6 = st.columns(3)


with P4:
    risk_reward_variable = st.slider(" Risk & Reward ", min_value=1, max_value=30, value=(2,5))

    

with P5:
    var_pop = st.slider("Problity of Profit %", min_value=10, max_value=100, value=(70,85))





def get_option_chain(ticker):
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
    response = requests.get(url)
    data = response.json()
    options_data = data['data']['options']
    df = pd.json_normalize(options_data)
    df['expiry'] = df['option'].str[-15:-9]
    df['type']  = df['option'].str[-9:-8]
    df['expiry'] = pd.to_datetime(df['expiry'], format='%y%m%d')
    
    df['days_to_expiry'] = (df['expiry'] - pd.Timestamp.today()).dt.days
    
    #df['abs_30_expiry'] = abs(30- df['days_to_expiry'])
    #min_30_exp = min(df['abs_30_expiry'])
    
    df['diff'] = abs((df['expiry'] - pd.Timestamp.today()).dt.days)
    df['diff'] = abs(dte-df['diff']) # added
    
    min_30_exp = min(df['diff']) # added
    df['strikes'] = df['option'].str[-8:]  # Get the last 8 characters
    df['left'] = df['strikes'].str[:5]     # Get the first 5 characters (significant digits)
    df['right'] = df['strikes'].str[-3:]   # Get the last 3 characters (fractional part)
    df['strikes'] = (df['left'] + df['right']).astype(float) / 1000  # Combine, convert to float, divide by 100
    df = df.drop(columns=['left', 'right'])
    min_diff = min(df['diff'])
    #filtered_df = df[df['diff'] == min_diff]
    filtered_df = df[df['diff'] == min_30_exp] #added
    max_expiry = max(filtered_df['expiry'])
    max_expiry = pd.to_datetime(max_expiry).date().strftime('%Y-%m-%d')
    latest_expiration = max_expiry
    option_chain = df[df['expiry'] == max_expiry]
    return option_chain, latest_expiration







for i in range(len(options)):
    ticker = options[i]
    option_chain = get_option_chain(ticker)[0]
    latest_expiration = get_option_chain(ticker)[1]
    option_chain = pd.DataFrame(option_chain)
    #st.write(option_chain)

    data_show = option_chain
    data_show_call = option_chain

    data_show = data_show[data_show['type'] == 'P']
    data_show_call = data_show_call[data_show_call['type'] == 'C']

    data_show_call['premium'] = (data_show_call['bid'] + data_show_call['ask']) / 2
    data_show['premium'] = (data_show['bid'] + data_show['ask']) / 2

    data_show['delta_20'] = abs(0.20 - abs(data_show['delta']))
    data_show_call['delta_20'] = abs(0.20 - abs(data_show_call['delta']))

    abs_20_delta = min(data_show['delta_20'])
    abs_20_delta_call = min(data_show_call['delta_20'])

    data_show = data_show[['option','strikes','delta','premium','iv','open_interest','delta_20']]
    data_show_call = data_show_call[['option','strikes','delta','premium','iv','open_interest','delta_20']]

    data_show_20_delta_put = data_show[data_show['delta_20'] == abs_20_delta]
    data_show_20_delta_call = data_show_call[data_show_call['delta_20'] == abs_20_delta_call]
    
    delta_at_20_var =  data_show_20_delta_put['delta']
    delta_at_20_var_call =  data_show_20_delta_call['delta']

    data_show_other_than_20 = data_show[data_show['delta'] > delta_at_20_var.iloc[0] ]
    data_show_other_than_20_call = data_show_call[data_show_call['delta'] < delta_at_20_var_call.iloc[0] ]

    data_show_other_than_20 = data_show_other_than_20.add_suffix('_temp')
    data_show_other_than_20_call = data_show_other_than_20_call.add_suffix('_buy')
    
    df_com      = data_show_20_delta_put.merge(data_show_other_than_20, how = 'cross')
    df_com_call = data_show_20_delta_call.merge(data_show_other_than_20_call, how = 'cross')


    df_com['risk'] = (df_com['strikes'] -  df_com['strikes_temp'])*100 - (df_com['premium'] - df_com['premium_temp'])*100
    df_com['reward'] = (df_com['premium'] - df_com['premium_temp'])*100
    
    df_com['ratio'] = ((df_com['strikes'] -  df_com['strikes_temp'])*100 - (df_com['premium'] - df_com['premium_temp'])*100)/ ((df_com['premium'] - df_com['premium_temp'])*100)

    df_com_call['risk'] = (df_com_call['strikes_buy'] -  df_com_call['strikes'])*100 - (df_com_call['premium'] - df_com_call['premium_buy'])*100
    df_com_call['reward'] = (df_com_call['premium'] - df_com_call['premium_buy'])*100

    df_com_call['ratio'] = ((df_com_call['strikes_buy'] -  df_com_call['strikes'])*100 - (df_com_call['premium']-df_com_call['premium_buy'])*100)//((df_com_call['premium'] - df_com_call['premium_buy'])*100) 



    
    bear_call_spread = df_com_call.rename(columns={
    'option': 'option_sell_call',
    'strikes': 'strikes_sell_call',
    'delta': 'delta_sell_call',
    'premium': 'premium_sell_call',
    'iv': 'iv_sell_cal',
    'open_interest': 'open_interest_sell_call',
    'delta_20': 'delta_20_sell_call',
    'option_buy': 'option_buy_call',
    'strikes_buy': 'strikes_buy_call',
    'delta_buy': 'delta_buy_call',
    'premium_buy': 'premium_buy_call',
    'iv_buy': 'iv_buy_call',
     'open_interest': 'open_interest_buy_call',
    'delta_20': 'delta_20_buy_call',
    'risk': 'risk_bear_call_spread',
    'reward':'reward_bear_call_spread',
    'ratio' : 'ratio_bear_call_spread'
    })
           
    bull_put_spread = df_com.rename(columns={
    'option': 'option_sell_put',
    'strikes': 'strikes_sell_put',
    'premium': 'premium_sell_put',
    'delta': 'delta_sell_put',
    
    'option_temp': 'option_buy_put',
    'strikes_temp': 'strikes_buy_put',
    'premium_temp': 'premium_buy_put',
    'delta_temp': 'delta_buy_put',
    'risk' : 'risk_bull_put_spread',
    'reward':'reward_bull_put_spread',
    'ratio' : 'ratio_bull_put_spread'
    })
            
  
    
    iron_codor_df =  bear_call_spread.merge(bull_put_spread, how = 'cross')
    iron_codor_df['REWARD'] = (iron_codor_df['premium_sell_call'] + iron_codor_df['premium_sell_put'] - iron_codor_df['premium_buy_call'] - iron_codor_df['premium_buy_put'])*100
    
    iron_codor_df['RISK'] = (np.maximum(iron_codor_df['strikes_buy_call'] - iron_codor_df['strikes_sell_call'],iron_codor_df['strikes_sell_put'] - iron_codor_df['strikes_buy_put']) - (iron_codor_df['premium_sell_call'] + iron_codor_df['premium_sell_put'] - iron_codor_df['premium_buy_call'] - iron_codor_df['premium_buy_put']))*100
    
    iron_codor_df['RISK & REWARD'] = (np.maximum(iron_codor_df['strikes_buy_call'] - iron_codor_df['strikes_sell_call'],iron_codor_df['strikes_sell_put'] - iron_codor_df['strikes_buy_put']) - (iron_codor_df['premium_sell_call'] + iron_codor_df['premium_sell_put'] - iron_codor_df['premium_buy_call'] - iron_codor_df['premium_buy_put'])) / (iron_codor_df['premium_sell_call'] + iron_codor_df['premium_sell_put'] - iron_codor_df['premium_buy_call'] - iron_codor_df['premium_buy_put']
)
    iron_codor_df['PROB PROFIT %'] = (1-abs(iron_codor_df['delta_sell_put']))*100
    
    iron_codor_df = iron_codor_df[ (iron_codor_df['RISK & REWARD'] >= risk_reward_variable[0]) & (iron_codor_df['RISK & REWARD'] <= risk_reward_variable[1]) ]

    iron_codor_df = iron_codor_df[ (iron_codor_df['PROB PROFIT %'] >= var_pop[0]) & (iron_codor_df['PROB PROFIT %'] <= var_pop[1])]

    sort_colmns = ['RISK', 'REWARD' , 'RISK & REWARD' , 'PROB PROFIT %' ]
    new_order = sort_colmns + [col for col in iron_codor_df.columns if col not in sort_colmns]
    iron_codor_df = iron_codor_df[new_order]
    iron_codor_df.columns = iron_codor_df.columns.str.upper()

    

    if strategy == 'Iron Condor':
        st.markdown( f""" <h1 style='color: #0ff550;'>Top Iron Condor Combination For : {options[i]}</h1> """, unsafe_allow_html=True)
        st.write(iron_codor_df)
    elif strategy == 'Bull Put Spread':
        st.markdown( f""" <h1 style='color: #0ff550;'>Top Bull Put Spreads Combination For : {options[i]}</h1> """, unsafe_allow_html=True)
        st.write(bull_put_spread)
    elif strategy == 'Bear Call Spread':
        st.markdown( f""" <h1 style='color: #0ff550;'>Top Bear Call Spreads Combination For : {options[i]}</h1> """, unsafe_allow_html=True)
        st.write('Bear Call Spread')
    else:
        st.write('Data Not Avaialble')
    





st.write("")
st.title("")
st.title("")
st.title("")
st.title("")
st.title("")
st.title("")
st.title("")
st.title("")

st.markdown(
    """
    <h4 style='color: #FFFFFF; text-align: center;'>Developed & Designed by Rishav Kant</h4>
    """, 
    unsafe_allow_html=True
)


#------------------------ Iron Condor Screener Section End -----------------------#


