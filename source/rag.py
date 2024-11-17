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


def init_vectorstore():
    # Load the data
    if os.path.isfile("./db/vectorstore"):
        with open("./db/vectorstore", "rb") as f:
            return FAISS.load_local(f)
    else:
        loader = TextLoader("../data/processed_wiki_ko.txt")
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
        vectorstore.save_local("./db/vectorstore")
        return vectorstore


def init_bm25_retriever():
    loader = TextLoader("../data/processed_wiki_ko.txt")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50, length_function=len)
    splits = text_splitter.split_documents(data)

    bm25_retriever = BM25Retriever.from_documents(splits)
    with open("./db/bm25_retriever/bm25.bin", "wb") as f:
        pickle.dump(bm25_retriever, f)

    return bm25_retriever


def retrieve_query(query: str, vectorstore=None):

    # if os.path.isfile('./db/bm25_retriever/bm25.bin'):
    #     print("Loading bm25_retriever")
    #     with open('./db/bm25_retriever/bm25.bin', 'rb') as f:
    #         bm25_retriever = pickle.load(f)
    # else:
    #     bm25_retriever = init_bm25_retriever()
    store = InMemoryStore()
    # child_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=50,
    #     chunk_overlap=50
    # )

    # # Parent splitter 설정 (반환용 더 큰 청크)
    # parent_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,  # 원하는 반환 길이
    #     chunk_overlap=50
    # )

    # parent_faiss = ParentDocumentRetriever(
    #     vectorstore=vectorstore,  # 기존 FAISS Retriever
    #     docstore=store,
    #     child_splitter=child_splitter,
    #     parent_splitter=parent_splitter,
    # )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # ensembel_retriever = EnsembleRetriever(
    #     retrievers = [bm25_retriever, parent_faiss], weights=[0.5, 0.5]
    # )

    # docs = ensembel_retriever.invoke(query)
    # docs = parent_faiss.invoke(query)
    # ret_docs = []
    # for doc in docs:
    #     ret_docs.append(doc.page_content)
    ret_docs = retriever.invoke(query)

    return ret_docs


if __name__ == "__main__":
    # id - 100
    query = "○장수왕은 남 진 정책의 일환으로 수도를 이곳으로 천도 하였다. ○묘청은 이곳으로 수도를 옮길 것을 주장하였다.'question': '밑줄 친 ‘이곳’에 대한 설명으로 옳은 것은?"
    ret_docs = retrieve_query(query)
    print(ret_docs)
    print("=" * 50)
    print(ret_docs[0].page_content)
