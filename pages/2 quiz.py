import json
from tkinter import NO
import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser

@st.cache_data(show_spinner="Splitting file...") # cache if the input is same
def split_file(file): # don't embed and use vectorstore because we just want to search (we just make a quiz using the whole contents in a file)
    file_dir = f"./.cache/quiz_files/{file.name}"
    file_content = file.read()
    with open(file_dir, "wb") as f:
        f.write(file_content)
    loader = UnstructuredFileLoader(file_dir)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    chunks = loader.load_and_split(text_splitter=splitter)
    return chunks

def combine_chunks(chunks): # combine a list of chunks into a single string
    return "\n\n".join(chunk.page_content for chunk in chunks)

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```json", "").replace("```", "")
        return json.loads(text)

@st.cache_data(show_spinner="Invoking final chain...")
def invoke_final_chain(_chunks, topic): 
    # final_chain = {"context": quiz_chain} | formatting_chain | parser
    final_chain = {"chunks": combine_chunks} | prompt | llm
    return final_chain.invoke(_chunks)

@st.cache_data(show_spinner="Searching Wikipedia...")
def search_wikipedia(topic):
    retriever = WikipediaRetriever(top_k_results=1) # return chunks relevant to topic from Wikipedia vectorstore
    return retriever.get_relevant_documents(topic)

st.set_page_config(
    page_title="Quiz GPT",
    page_icon="❓",
)

st.title("Quiz GPT")
    
# stuff_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a helpful assistant that is role playing as a teacher.
                
#             Based ONLY on the following chunks make 10 questions to test the user's knowledge about the text.
            
#             Each question should have 4 answers, three of them must be incorrect and one should be correct.
                
#             Use (o) to signal the correct answer.
                
#             Question examples:
                
#             Question: What is the color of the ocean?
#             Answers: Red|Yellow|Green|Blue(o)
                
#             Question: What is the capital or Georgia?
#             Answers: Baku|Tbilisi(o)|Manila|Beirut
                
#             Question: When was Avatar released?
#             Answers: 2007|2001|2009(o)|1998
                
#             Question: Who was Julius Caesar?
#             Answers: A Roman Emperor(o)|Painter|Actor|Model
                
#             Your turn!
                
#             Chunks: {chunks}
#             """
#         )
#     ]
# )

# formatting_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a powerful formatting algorithm.
            
#             You format exam questions into JSON format.
#             Answers with (o) are the correct ones.
            
#             Example Input:

#             Question: What is the color of the ocean?
#             Answers: Red|Yellow|Green|Blue(o)
                
#             Question: What is the capital or Georgia?
#             Answers: Baku|Tbilisi(o)|Manila|Beirut
                
#             Question: When was Avatar released?
#             Answers: 2007|2001|2009(o)|1998
                
#             Question: Who was Julius Caesar?
#             Answers: A Roman Emperor(o)|Painter|Actor|Model
            
            
#             Example Output:
#             ```json
#             {{ "questions": [
#                     {{
#                         "question": "What is the color of the ocean?",
#                         "answers": [
#                                 {{
#                                     "answer": "Red",
#                                     "correct": false
#                                 }},
#                                 {{
#                                     "answer": "Yellow",
#                                     "correct": false
#                                 }},
#                                 {{
#                                     "answer": "Green",
#                                     "correct": false
#                                 }},
#                                 {{
#                                     "answer": "Blue",
#                                     "correct": true
#                                 }},
#                         ]
#                     }},
#                                 {{
#                         "question": "What is the capital or Georgia?",
#                         "answers": [
#                                 {{
#                                     "answer": "Baku",
#                                     "correct": false
#                                 }},
#                                 {{
#                                     "answer": "Tbilisi",
#                                     "correct": true
#                                 }},
#                                 {{
#                                     "answer": "Manila",
#                                     "correct": false
#                                 }},
#                                 {{
#                                     "answer": "Beirut",
#                                     "correct": false
#                                 }},
#                         ]
#                     }},
#                                 {{
#                         "question": "When was Avatar released?",
#                         "answers": [
#                                 {{
#                                     "answer": "2007",
#                                     "correct": false
#                                 }},
#                                 {{
#                                     "answer": "2001",
#                                     "correct": false
#                                 }},
#                                 {{
#                                     "answer": "2009",
#                                     "correct": true
#                                 }},
#                                 {{
#                                     "answer": "1998",
#                                     "correct": false
#                                 }},
#                         ]
#                     }},
#                     {{
#                         "question": "Who was Julius Caesar?",
#                         "answers": [
#                                 {{
#                                     "answer": "A Roman Emperor",
#                                     "correct": true
#                                 }},
#                                 {{
#                                     "answer": "Painter",
#                                     "correct": false
#                                 }},
#                                 {{
#                                     "answer": "Actor",
#                                     "correct": false
#                                 }},
#                                 {{
#                                     "answer": "Model",
#                                     "correct": false
#                                 }},
#                         ]
#                     }}
#                 ]
#             }}
#             ```

#             Your turn!

#             {context}

#             """,
#         )
#     ]
# )
# quiz_chain = {"chunks": combine_chunks} | stuff_prompt | llm
# formatting_chain = formatting_prompt | llm
# parser = JsonOutputParser()

function = {
    "name": "create_quiz", 
    "description": "function that takes a list of questions and answers and returns a quiz", 
    "parameters": { 
        "type": "object",
        "properties": { 
            "questions": { 
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": { 
                        "question": { 
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

prompt = PromptTemplate.from_template("Make a quiz with {chunks}")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4.1-nano",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={"name": "create_quiz"},
    functions=[function],
)

with st.sidebar:
    chunks = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia",
        ),
    )
    if choice == "File":
        file = st.file_uploader("Upload a file")
        if file:
            chunks = split_file(file)
    elif choice == "Wikipedia":
        topic = st.text_input("Search Wikipedia...")
        if topic:
            chunks = search_wikipedia(topic)

if chunks:
    # answer = chain.invoke(chunks)
    # formatted_answer = formatting_chain.invoke({"context": answer.content})
    final_answer = invoke_final_chain(chunks, topic if topic else file.name)
    final_answer = final_answer.additional_kwargs["function_call"]["arguments"]
    final_answer = json.loads(final_answer) 

    with st.form("quiz"):
        for question in final_answer["questions"]:
            st.write(question["question"])
            selected_answer = st.radio(
                label="Select an answer",
                options=[answer["answer"] for answer in question["answers"]],
                index=None
            )
            if {"answer": selected_answer, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif selected_answer is not None:
                st.error("Incorrect!")
        submit_button = st.form_submit_button("Submit")
        

else:
    st.markdown(
        """
        Make a quiz from Wikipedia articles or files you upload to test your knowledge!
        """
    )
