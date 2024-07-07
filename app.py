import os
import ast
import re
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables import RunnablePassthrough
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import Optional
from system import system
from loguru import logger
from models import Query
from time import perf_counter
from functools import lru_cache

# Load environment variables from .env file
load_dotenv()

# just defining the OPENAI_API_KEY will help us authenticate with the openapi api
# Set up API key authentication
# API_KEY = os.getenv("OPENAI_API_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
api_key_header = APIKeyHeader(name="ACCESS_TOKEN", auto_error=False)


def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == ACCESS_TOKEN:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


# Initialize FastAPI app
app = FastAPI()


def create_conn_string(db: str) -> str:
    """Generate the db string based on the database type

    Args:
        db (str): database type like mysql, postgres, etc

    Returns:
        str: connection string
    """
    # Database connection details
    HOST = os.environ.get("DB_HOST")
    USERNAME = os.environ.get("DB_USER")
    PASSWORD = os.environ.get("DB_PASSWORD")
    DATABASE = os.environ.get("DB_NAME")

    if db.lower() == "mysql":
        return f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"
    elif "postgres" in db.lower():  # postgres can be given as postgresql also
        return f"postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"
    else:
        logger.exception("Not Supported Database, only supports postgres and mysql")
        raise Exception("Not Supported DataBase")


@lru_cache
def get_database_engine() -> SQLDatabase:
    # Construct the connection URI
    connection_uri = create_conn_string(os.environ.get("DB_TYPE"))

    # Initialize the database using the connection URI
    db = SQLDatabase.from_uri(connection_uri, lazy_table_reflection=True)
    # db = SQLDatabase.from_uri(connection_uri)

    return db


@lru_cache
def get_agent(db: str = Depends(get_database_engine)) -> CompiledGraph:
    llm = ChatOpenAI(model="gpt-4o")

    # Create the agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    system_message = SystemMessage(
        content=system.format(table_names=db.get_usable_table_names())
    )

    agent = create_react_agent(llm, tools, messages_modifier=system_message)

    return agent


# Define Pydantic model
@app.get("/")
async def hello(api_key: str = Depends(get_api_key)):
    return {"msg": "Hello"}


# Define endpoint
@app.post("/query")
async def query_agent(
    request: Query,
    agent: CompiledGraph = Depends(get_agent),
    api_key: str = Depends(get_api_key),
):
    start = perf_counter()
    human_message = HumanMessage(content=request.question)
    response = []
    for s in agent.stream({"messages": [human_message]}):
        response.append(s)
    return {"response": response, "time_taken": perf_counter() - start}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
