import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from config import settings


# loading env vars
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("API KEY not defined")


# model initialize
model = init_chat_model(model=settings.model, temperature=0.6)


# load pdf
pdf_path = Path(__file__).parent.parent / "data" / "attention.pdf"
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

print(len(docs))

#split and chunk doc
splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
chunks = splitter.split_documents(docs)
print(len(chunks))

#Embedding
embedding = OpenAIEmbeddings(
    model=settings.embedding_model
)

#Vectorization
texts = [chunk.content for chunk in chunks]
vector = embedding.embed_documents(texts)
print(vector)




def generate_terms_and_jokes(context):
    prompt_comedy_about = ChatPromptTemplate.from_messages([
        ("system" ,"You are a comedian, I will first give you a topic, you will have to produce some terms, but maximum 3. And each of these terms, should be one single word"),
        ("user" , "generate for  {context}")
    ])
    prompt_comedy_on_terms = ChatPromptTemplate.from_template("tell me a joke on these terms : {terms}")
    about_chain = prompt_comedy_about | model
    terms = about_chain.invoke(context)
    term_chain = prompt_comedy_on_terms | model
    response = term_chain.invoke({"terms" : terms.content})
    return terms.content, response.content