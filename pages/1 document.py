# from langchain.embeddings import OllamaEmbeddings
# from langchain.chat_models import ChatOllama
# OllamaEmbeddings(model="deepseek-r1")
# ChatOllama(model="deepseek-r1")

import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory

@st.cache_data(
    show_spinner="Retrieving file..."
)  # only if file is same, it won't run the function again
def retrieve_file(file):
    file_dir = f"./.cache/document_files/{file.name}"
    embedding_dir = LocalFileStore(f"./.cache/document_embeddings/{file.name}")
    embeddings = OpenAIEmbeddings()

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
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, embedding_dir
    )
    vectorstore = FAISS.from_documents(chunks, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append(
        {"message": message, "role": role}
    )  # Streamlit reruns the entire script from top to bottom on every interaction! So, we need to use session state to store the messages.

def show_message(message, role, save=True):
    with st.chat_message(
        role
    ):  # if use 'with', we can automatically use the functions inside chat_message
        st.markdown(message)  # print the message
    if save:
        save_message(message, role)

def load_messages():
    for message in st.session_state["messages"]:
        show_message(message["message"], message["role"], save=False)

def combine_chunks(chunks):
    return "\n\n".join(chunk.page_content for chunk in chunks)


class LLMCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, *args, **kwargs):  # when llm starts generating new tokens
        self.message_box = st.empty()  # empty widget to update the message
        self.message = ""

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

st.set_page_config(
    page_title="Document GPT",
    page_icon="📃",
)

st.title("Document GPT")

st.markdown(
    """
    Use this chatbot to ask questions to an AI about your files!
    """
)

# stuff_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a helpful assistant. 
#             Answer user message using the following chunks.
            
#             chunks: {chunks}
#             """,
#         ),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{user_message}"),
#     ]
# )

get_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. 
    Using the chat history and chunk to answer the user message.                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user message the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.

    Chat histroy: {history}
    Chunk: {chunk}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    User message: {user_message}
    """
)

def get_answers(inputs):
    history = inputs["history"]
    chunks = inputs["chunks"]
    user_message = inputs["user_message"]
    get_chain = get_prompt | ChatOpenAI(temperature=0.1)
    return {
        "user_message": user_message,
        "answers": [
            {
                "answer": get_chain.invoke(
                    {"history": history , "chunk": chunk, "user_message": user_message}
                ).content,
            }
            for chunk in chunks
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's message.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Answers: {combined_answers}
            """,
        ),
        ("human", "{user_message}"),
    ]
)

def choose_answer(inputs):
    user_message = inputs["user_message"]
    answers = inputs["answers"]
    combined_answers = "\n\n".join(
        answer['answer'] for answer in answers
    )
    choose_chain = choose_prompt | ChatOpenAI(
                                    temperature=0.1,
                                    streaming=True,
                                    callbacks=[LLMCallbackHandler()])  # callbacks are called when chain is invoked
    return choose_chain.invoke(
        {
            "user_message": user_message,
            "combined_answers": combined_answers,
        }
    )

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(), return_messages=True
    )
memory = st.session_state["memory"]

with st.sidebar:
    file = st.file_uploader("Upload a file")

if file:
    retriever = retrieve_file(file)

    show_message("File loaded successfully!", "ai", save=False)  # first ai message
    load_messages()  # load previous messages

    user_message = st.chat_input("Ask a question about your file")
    if user_message:
        show_message(user_message, "human")

        # chunks = retriever.invoke(user_message) # return chunks
        # combined_chunks = "\n\n".join(chunk.page_content for chunk in chunks)
        # prompt = stuff_template.format_messages(combined_chunks=combined_chunks, question=user_message)
        # llm.predict_messages(prompt)

        map_rerank_chain = (
            {
                "history": RunnableLambda(lambda _: memory.load_memory_variables({})["history"]),
                "chunks": retriever, # | RunnableLambda(combine_chunks),
                "user_message": RunnablePassthrough(),
            }
            # | stuff_prompt
            # | llm
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            ai_message = map_rerank_chain.invoke(
                user_message
            )  # user_message is passed to retriever and question

        memory.save_context(
            {"input": user_message},
            {"output": ai_message.content},
        )

else:
    st.session_state["messages"] = []
    st.session_state["memory"].clear()