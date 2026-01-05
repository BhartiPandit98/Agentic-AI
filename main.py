from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import chat_models
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str=Field(description="the URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOpenAI(model="gpt-5")
tools = [TavilySearch(
    max_results=5
)]
agent = create_agent(model=llm,tools=tools,response_format=AgentResponse)

def main():
    print("Hello from langchain")
    result = agent.invoke({"messages":HumanMessage(content="Give me list of job openings posted within 24 hours on Linkedin for agentic ai")})
    print(result)

if __name__ == "__main__":
    main()
