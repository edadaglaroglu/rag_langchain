from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get the OpenAI API key from environment variables
my_key_openai = os.getenv("openai_apikey")

embeddings = OpenAIEmbeddings(api_key=my_key_openai)

llm_openai = ChatOpenAI(api_key=my_key_openai, model="gpt-4o")

# embeddings = OpenAIEmbeddings(api_key=my_key_openai)
# embeddings = CohereEmbeddings(cohere_api_key=my_key_cohere, model="embed-multilingual-v3.0") #embed-english-v3.0

def ask_openai(prompt):
    AI_Response = llm_openai.invoke(prompt)
    return AI_Response.content

def rag_with_url(target_url, prompt):
    loader = WebBaseLoader(target_url)

    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )

    splitted_documents = text_splitter.split_documents(raw_documents)

    vectorstore = FAISS.from_documents(splitted_documents, embeddings)
    retriever = vectorstore.as_retriever()

    relevant_documents = retriever.get_relevant_documents(prompt)

    context_data = ""

    for document in relevant_documents:
        context_data = context_data + " " + document.page_content

    final_prompt = f"""I have a question: {prompt}
    To answer this question, we have the following information: {context_data} .
    Use only the information provided here to answer the question. Do not go beyond these.
    """

    AI_Response = llm_openai.invoke(final_prompt)

    return AI_Response.content

def rag_with_pdf(filepath, prompt):
    loader = PyPDFLoader(filepath)

    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )

    splitted_documents = text_splitter.split_documents(raw_documents)

    vectorstore = FAISS.from_documents(splitted_documents, embeddings)
    retriever = vectorstore.as_retriever()

    relevant_documents = retriever.get_relevant_documents(prompt)

    context_data = ""

    for document in relevant_documents:
        context_data = context_data + " " + document.page_content

    final_prompt = f"""I have a question: {prompt}
    To answer this question, we have the following information: {context_data} .
    Use only the information provided here to answer the question. Do not go beyond these.
    """

    AI_Response = llm_openai.invoke(final_prompt)

    return AI_Response.content, relevant_documents
