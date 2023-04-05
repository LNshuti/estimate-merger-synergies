from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from pinecone import PineconeClient
from langchain.prompts import load_prompt
import pinecone
from langchain import OpenAI, ConversationChain
import openai
import os
from unstructured.partition.auto import partition

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize the Pinecone client with a project name
#pinecone_client = PineconeClient()

OPENAI_API_KEY = "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"
PINECONE_API_KEY = '82f63bcf-312f-4b33-a8c4-89c9997c5016'
PINECONE_API_ENV = 'us-central1-gcp'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchain2"

openai.api_key = "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"

llm = OpenAI(temperature=0, openai_api_key=openai.api_key)
conversation = ConversationChain(llm=llm, verbose=True)

pinecone.init(api_key=openai.api_key, environment="us-west1-gcp")
loader = UnstructuredPDFLoader("/content/drive/MyDrive/Medical Records/Leonce_Nshuti_full_medical_records.pdf")

data = loader.load()
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print (f'Now you have {len(texts)} documents')

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

query = "Using the available information, outline a plan to improve key health indicators in the records. Use generalizable advice as a nutritionist and wellness coach. Provide details for a 6 months plan. Use bullet points."
docs = docsearch.similarity_search(query, include_metadata=True)

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
print(chain.run(input_documents=docs, question=query))
