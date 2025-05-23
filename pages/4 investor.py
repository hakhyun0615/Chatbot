from urllib import response
from langchain.schema import SystemMessage
import streamlit as st
import os
import requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")

class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for. Example query: Stock Market Symbol for Apple Company")

class StockMarketSymbolSearchTool(BaseTool):
    name = "StockMarketSymbolSearchTool"
    description = """
    Use this tool to find the stock market symbol for a company.
    It takes a query as an argument.
    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)

class CompanyArgsSchema(BaseModel):
    symbol: str = Field(description="Stock symbol of the company.Example: AAPL,TSLA")

class CompanyOverviewTool(BaseTool):
    name = "CompanyOverview"
    description = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyArgsSchema] = CompanyArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()

class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatement"
    description = """
    Use this to get the income statement of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyArgsSchema] = CompanyArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()["annualReports"]

class CompanyStockPerformanceTool(BaseTool):
    name = "CompanyStockPerformance"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyArgsSchema] = CompanyArgsSchema

    def _run(self, symbol):
        r = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}")
        response = r.json()
        return list(response["Weekly Time Series"].items())[:200]

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="📈",
)

st.markdown(
    """
    # Investor GPT
                
    Write down the name of a company and our Agent will do the research for you.
    """
)

company = st.text_input("Write the name of the company you are interested in.")

agent = initialize_agent(
    llm=ChatOpenAI(temperature=0.1, model="gpt-4.1-mini"),    
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    verbose=True,
    tools=[
        StockMarketSymbolSearchTool(), # search for stock symbol
        CompanyOverviewTool(), # get company overview using the stock symbol
        CompanyIncomeStatementTool(), # get company income statement using the stock symbol
        CompanyStockPerformanceTool() # get company stock performance using the stock symbol
    ],
    agent_kwargs={
        "system_message": SystemMessage( # change the system prompt (OPENAI_FUNCTIONS's default system prompt: "you are a helpful AI assistant") instead of using human message 
            content="""
            You are a hedge fund manager.
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
            Consider the performance of a stock, the company overview and the income statement.
            Be assertive in your judgement and recommend the stock or advise the user against it.
            """
        )
    },
)

if company:
    answer = agent.invoke(company)
    st.write(answer["output"].replace("$", "\$"))