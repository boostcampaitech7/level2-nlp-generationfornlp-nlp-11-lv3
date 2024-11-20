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


def init_vectorstore(vectorstore_path):
    # Load the data
    if os.path.isfile(vectorstore_path):
        with open(vectorstore_path, "rb") as f:
            return FAISS.load_local(f)
    else:
        wiki_path = os.path.join(os.path.dirname(__file__), "../data/processed_wiki_ko.txt")
        loader = TextLoader(wiki_path)
        data = loader.load()

        # Split the data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=100, length_function=len)
        splits = text_splitter.split_documents(data)

        embeddings_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sbert-nli",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = FAISS.from_documents(
            documents=splits, embedding=embeddings_model, distance_strategy=DistanceStrategy.COSINE
        )
        vectorstore.save_local(vectorstore_path)
        return vectorstore


def init_bm25_retriever():
    wiki_path = os.path.join(os.path.dirname(__file__), "../data/processed_wiki_ko.txt")
    loader = TextLoader(wiki_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50, length_function=len)
    splits = text_splitter.split_documents(data)

    bm25_retriever = BM25Retriever.from_documents(splits)
    with open("./db/bm25_retriever/bm25.bin", "wb") as f:
        pickle.dump(bm25_retriever, f)

    return bm25_retriever


def retrieve_query(query: str, vectorstore=None):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    ret_docs = retriever.invoke(query)

    return ret_docs


if __name__ == "__main__":
    embeddings_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db_path = os.path.join(os.path.dirname(__file__),"../db/vectorstore")
    if os.path.exists(db_path):
        print("Loading vectorstore")
        vectorstore = FAISS.load_local("./db/vectorstore", embeddings_model, allow_dangerous_deserialization=True)
    else:
        vectorstore = init_vectorstore()
    # id - 100
    query = "○불교를 수용하였다. ○태학을 설립하였다.'question': '다음 정책을 시행한 국왕의 재위 기간에 있었던 사실로 옳은 것은?"
    ret_docs = retrieve_query(query, vectorstore)
    print(ret_docs)
    print("=" * 50)
    print(ret_docs[0].page_content)
