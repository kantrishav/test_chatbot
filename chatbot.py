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
        <h1 style='color: #8c8c8c; font-size: 20px;'>Choose the best option strategy for you. Just paste your option chain data.</h1>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color: #0ff550; font-size: 20px;'>Drop Your Option Chain in the Box and hit Submit !</h1>
    </div>
    """,
    unsafe_allow_html=True
)



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



option_text_input = st.text_area('', height=310)

option_prompt = "You are an AI Option trader. Below is the text format of Option Chain of an asset. Analyze and share top 3 strategies using option chain data uploaded and share how can we build that strategies with the data shared.  Here is the option chain text : "

option_final_prompt = option_prompt + option_text_input


#with st.container():
 #   a, b, c = st.columns([0.49, 1, 1])  # Adjust the width ratios as needed
  #  with b:  # This places it in the center column
   #     st.button("Submit",key="submit_button_1")




# Add the icons in the bottom right corner
st.markdown(
    """
    <div class="icon-container">
        <img src="https://thevyatergroup.com/wp-content/uploads/2021/03/logo-amazon-404px-grey.png" alt="Icon 1">
        <img src="https://www.krenerbookkeeping.com/wp-content/uploads/2018/07/logo-microsoft-404px-grey.png" alt="Icon 2">
        <img src="https://mohamadfaizal.com/wp-content/uploads/2017/05/logo-google-404px-grey.png" alt="Icon 3">

        
    </div>
    """,
    unsafe_allow_html=True
)






if len(option_final_prompt) > 500:
    chat_completion = client.chat.completions.create( messages=[{"role": "user","content": option_final_prompt,}],model="gpt-3.5-turbo",)
    op = chat_completion.choices[0].message.content
    op = op.replace('\n', '<br>')
    st.markdown(f'<p style="color:white;">{op}</p>', unsafe_allow_html=True)
    #st.markdown(f'<p style="color:white;">{op}</p>', unsafe_allow_html=True)
    #st.write(op)
else:
    st.write('')
 
#----------------------------------------------------FinChat---------------------------------------------

from phi.agent import Agent, Tool
from phi.tools import Toolkit
import requests
import pandas as pd

    
from phi.model.openai.chat import OpenAIChat
from secret_key import openapi_key  # Assuming 'secret_key.py' contains your OpenAI API key
import os
os.environ['OPENAI_API_KEY'] = openapi_key


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



query = st.text_input("Ask about options data for a stock:") 
query = (query + " Show the output in strctured format using tables")


if query:
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        response = agent.run(query)
        content = response.content
        st.write(content)


# #-----------------------------------------------------------------------------

#agent.print_response("Fetch the options data for MICROSOFT and give the top 3 iron condor strategy", stream=False)




