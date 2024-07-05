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
from langchain_core.runnables import RunnablePassthrough
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Set up API key authentication
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = os.getenv("ACCESS_TOKEN")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# Initialize FastAPI app
app = FastAPI()

# Initialize the database and LLM
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Database connection details
HOST = "3.139.130.189"
USERNAME = "hariom"
PASSWORD = "T22e66981D5NR"
DATABASE = "forge"

# Construct the connection URI
connection_uri = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"

# Initialize the database using the connection URI
db = SQLDatabase.from_uri(connection_uri)



llm = ChatOpenAI(model="gpt-4o")

# Create the agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()


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

system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

You have access to the following tables: {table_names}

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool!
Do not try to guess at the proper name - use this function to find similar ones.""".format(
    table_names=db.get_usable_table_names()
)

system_message = SystemMessage(content=system)

agent = create_react_agent(llm, tools, messages_modifier=system_message)

# Define Pydantic model
class Query(BaseModel):
    question: str

# Define endpoint
@app.post("/query")
async def query_agent(request: Query, api_key: str = Depends(get_api_key)):
    human_message = HumanMessage(content=request.question)
    response = []
    for s in agent.stream({"messages": [human_message]}):
        response.append(s)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)