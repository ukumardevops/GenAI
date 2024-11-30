from decouple import config
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
#from dotenv import load_dotenv
import os


app = FastAPI()

#load_dotenv()
#os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

openai_model = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=config("OPENAI_API_KEY"))
prompt = ChatPromptTemplate.from_template(
    "Give me a summary about {topic} in a paragraph or less.")

chain = prompt | openai_model

add_routes(app, chain, path="/openai")

if __name__ == "__main__":
    import uvicorn

    host = os.environ["HOST"]
    port = os.environ["PORT"]

    uvicorn.run(app, host=host, port=port)