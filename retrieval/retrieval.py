import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import settings
from constants import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES


load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("API KEY not defined")

# model initialize
chat = init_chat_model(model=settings.model, temperature=0.6)

def get_vector_store():
    embedding = OpenAIEmbeddings(model=settings.embedding_model)
    vector_store = Chroma(
        collection_name="basic_rag_pipeline",
        embedding_function=embedding,
        persist_directory="./chroma_db")
    return vector_store

def retrieve(query, k=settings.top_k):
    vector_store = get_vector_store()
    return vector_store.similarity_search(
        query,
        k=k
    )

def build_context(documents):
    return "\n\n".join(f"""[Citation: Page {doc.metadata.get('page')},Lines {doc.metadata.get('line_from')}-{doc.metadata.get('line_to')}]{doc.page_content}"""
                       for doc in documents)

def build_prompt():
    messages = [
        ("system", SYSTEM_PROMPT)
    ]
    for example in FEW_SHOT_EXAMPLES:
        messages.append(("user", example["question"]))
        messages.append(("assistant", example["answer"]))
    messages.append(("user", "{question}"))
    return ChatPromptTemplate.from_messages(messages)


def generate_results(question):
    docs = retrieve(question, 10)
    context = build_context(documents=docs)
    prompt = build_prompt()
    chain = prompt | chat

    response = chain.invoke({
        "question": question,
        "context": context
    })
    return response.content
            
            


