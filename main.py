import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

"""
used to make server focused on making APIs

local server: uvicorn main:app --reload
    http://127.0.0.1:8000/docs
    http://127.0.0.1:8000/openapi.json -> used in GPT Action
public server: cloudflared tunnel --url http://127.0.0.1:8000
    https://russia-nasa-brunette-chains.trycloudflare.com
    https://surprised-fashion-removal-microphone.trycloudflare.com
"""

# app = FastAPI(
#     title="Nicolacus Maximus Quote Giver",
#     description="Get a real quote by Nicolacus Maximus",
#     servers=[{"url":"https://russia-nasa-brunette-chains.trycloudflare.com"}]
# )

# class Quote(BaseModel):
#     quote: str = Field(description="The quote that Nicolacus Maximus said.")
#     year: int = Field(description="The year when Nicolacus Maximus said the quote.")

# @app.get(
#     "/quote",
#     summary="Returns a random quote by Nicolacus Maximus",
#     description="Upon receiving a GET request this endpoint will return a real quiote said by Nicolacus Maximus himself.",
#     response_description="A Quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said.",
#     response_model=Quote,
# ) # http://127.0.0.1:8000/quote
# def get_quote():
#     return {
#         "quote": "Life is short so eat it all.",
#         "year": 1950,
#     }

load_dotenv() # load .env file

Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore.from_existing_index("recipes", embeddings)

app = FastAPI(
    title="CheftGPT. The best provider of Indian Recipes in the world.",
    description="Give ChefGPT the name of an ingredient and it will give you multiple recipes to use that ingredient on in return.",
    servers=[{"url":"https://surprised-fashion-removal-microphone.trycloudflare.com"}] # cloudflared tunnel --url http://127.0.0.1:8000
)

class Document(BaseModel):
    page_content: str

@app.get( # /recipes 라우트에 HTTP GET 요청이 들어올 때 실행
    "/recipes",
    summary="Returns a list of recipes.",
    description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that ingredient.",
    response_description="A Document object that contains the recipe and preparation instructions.",
    response_model=list[Document],
    openapi_extra={"x-openai-isConsequential": False},
) # http://127.0.0.1:8000/recipes
def get_recipe(ingredient: str):
    docs = vectorstore.similarity_search(ingredient)
    return docs