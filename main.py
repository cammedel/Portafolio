from pinecone import Pinecone
import os

os.environ['PINECONE_API_KEY'] = "pcsk_5UfTv5_7ZAS51WAGBAoKgBRWPZuQfFFtp6qJPvMvUmRmebMGEmTiY1ebH2uHFFBMHbU6R8"
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Lista los índices disponibles
print(pc.list_indexes().names())


#describir el index
INDEX_NAME="portafolio"
index = pc.Index(INDEX_NAME)
#index_description = pc.describe_index(index_name)
#print(index_description)

#importa la libreria (vectores y matrices), también se crea los vectores y sus valores
import numpy as np 
vector_1 = np.random.uniform(-1,1,384).tolist()
vector_2 = np.random.uniform(-1,1,384).tolist()

#se le otorgan bases a los vectores 
upsert_response = index.upsert(
    vectors=[ {'id': "vec1", "values": vector_1, "metadata": {'genre': 'cine'}},
        {'id': "vec2", "values": vector_2, "metadata": {'genre': 'teatro'}},
    ]
)

#se buscaran los vectores
vector_pregunta = np.random.uniform(-1, 1, 384).tolist()

result = result = index.query(vector=[vector_pregunta], top_k=1)
print(result)

#pdf
from langchain_community.document_loaders import PyPDFLoader

FILE = "la-rosaleda_v3.pdf"
loader = PyPDFLoader(FILE)
doc = loader.load()

#crear chuncks 
#ayuda para dividir los parrafos

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len
    )
chunks = text_splitter.split_documents(doc)

len(chunks)

def create_chunks(doc_to_chunk):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
        )
    return text_splitter.split_documents(doc_to_chunk)