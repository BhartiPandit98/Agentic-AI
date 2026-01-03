from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import chat_models
from langchain_tavily import TavilySearch

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch]
agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from langchain")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in tokyo?")})
    print(result)

if __name__ == "__main__":
    main()
