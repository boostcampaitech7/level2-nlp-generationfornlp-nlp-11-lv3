import os
import pickle

from langchain import LLMChain
from langchain.llms import huggingface_pipeline
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)

def init_vectorstore(vectorstore_path):

    embeddings_model = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sbert-nli',
            model_kwargs={'device':'cuda'},
    )

    # Load the data
    if  os.path.isdir(vectorstore_path):
        return FAISS.load_local(vectorstore_path, embeddings_model,allow_dangerous_deserialization=True)
    else:
        wiki_path = os.path.join(os.path.dirname(__file__), "../data/processed_wiki_ko.txt")
        loader = TextLoader(wiki_path)
        data = loader.load()

        # Split the data
        text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 500,
                    chunk_overlap = 100,
                    length_function = len
                )
        splits = text_splitter.split_documents(data)

        
        vectorstore = FAISS.from_documents(
            documents=splits, embedding=embeddings_model, distance_strategy=DistanceStrategy.COSINE
        )
        vectorstore.save_local(vectorstore_path)
        return vectorstore


def init_bm25_retriever():
    wiki_path = os.path.join(os.path.dirname(__file__), "../data/processed_wiki_ko.txt")
    loader = TextLoader(wiki_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1200,
            chunk_overlap = 100,
            length_function = len
        )    
    splits = text_splitter.split_documents(data)

    bm25_retriever = BM25Retriever.from_documents(splits)
    with open("./db/bm25_retriever/bm25.bin", "wb") as f:
        pickle.dump(bm25_retriever, f)

    return bm25_retriever


def retrieve_query(query: str, vectorstore=None):

    retriever = vectorstore.as_retriever(
            search_type = 'mmr',
            search_kwargs = {'k': 50, 'fetch_k':10, 'lambda_mult':0.6},
        )    
    ret_docs = retriever.invoke(query)

    return ret_docs

def simple_retrieve(query):
    embeddings_model = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sbert-nli',
            model_kwargs={'device':'cuda'},
            encode_kwargs={'normalize_embeddings':True},
        )
    db_path = os.path.join(os.path.dirname(__file__), "../db/vectorstore")
    vectorstore = init_vectorstore(db_path)
    ret_docs = retrieve_query(query, vectorstore)
    return ret_docs

def bm25_retrieve(query:str, bm25_retriever=None):
    bm25_retriever.k = 50
    docs = bm25_retriever.invoke(query)
    return docs

def re_ranking(docs,query,k=5):
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
    model.eval()
    pairs = []
    for doc in docs:
        pairs.append((query, doc.page_content))
    
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = exp_normalize(outputs.numpy())

    idx = np.argsort(scores)[::-1][:k]
    ret_docs = []
    for i in idx:
        ret_docs.append(docs[i])
    return ret_docs

def rerank_retrieve(query:str):
    docs = simple_retrieve(query)
    return re_ranking(docs,query)

if __name__ == "__main__":
    query = "○불교를 수용하였다. ○태학을 설립하였다.'question': '다음 정책을 시행한 국왕의 재위 기간에 있었던 사실로 옳은 것은?"
    ret_docs = retrieve_query(query)
    print(ret_docs)
    print("=" * 50)
    print(ret_docs[0].page_content)
