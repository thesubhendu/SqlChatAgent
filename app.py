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

# Load environment variables from .env file
load_dotenv()

# just defining the OPENAI_API_KEY will help us authenticate with the openapi api
# Set up API key authentication
API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY_NAME = os.getenv("ACCESS_TOKEN")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key_header: str = Depends(api_key_header)):
    print(api_key_header, API_KEY_NAME, API_KEY)
    if api_key_header == API_KEY:
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
        return f"postgresql+psycopg2://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    else:
        logger.exception("Not Supported Database, only supports postgres and mysql")
        raise Exception("Not Supported DataBase")


def get_database_engine(use_sqlite: Optional[bool] = False) -> SQLDatabase:
    if not use_sqlite:
        # Initialize the database and LLM
        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        return db

    # Construct the connection URI
    connection_uri = create_conn_string(os.environ.get("DB_TYPE"))

    # Initialize the database using the connection URI
    db = SQLDatabase.from_uri(connection_uri)

    return db


# def query_as_list(db, query):
#     res = db.run(query)
#     res = [el for sub in ast.literal_eval(res) for el in sub if el]
#     res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
#     return list(set(res))


# artists = query_as_list(db, "SELECT Name FROM Artist")
# albums = query_as_list(db, "SELECT Title FROM Album")


# vector_db = FAISS.from_texts(artists + albums, OpenAIEmbeddings())
# retriever = vector_db.as_retriever(search_kwargs={"k": 5})
# description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is valid proper nouns. Use the noun most similar to the search."""
# retriever_tool = create_retriever_tool(
#     retriever,
#     name="search_proper_nouns",
#     description=description,
# )

# tools.append(retriever_tool)


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


# Define endpoint
@app.post("/query")
async def query_agent(
    request: Query,
    agent: CompiledGraph = Depends(get_agent),
    api_key: str = Depends(get_api_key),
):
    human_message = HumanMessage(content=request.question)
    response = []
    for s in agent.stream({"messages": [human_message]}):
        response.append(s)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
