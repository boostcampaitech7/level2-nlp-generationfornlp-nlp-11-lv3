import os
import pickle

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import huggingface_pipeline
from langchain.storage import InMemoryStore
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ParentDocumentRetriever

def init_vectorstore():
    #Load the data
    if  os.path.isfile('./db/vectorstore'):
        with open('./db/vectorstore', 'rb') as f:
            return FAISS.load_local(f)
    else:
        loader = TextLoader('../data/processed_wiki_ko.txt')
        data = loader.load()

        #Split the data
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 150,
            chunk_overlap = 100,
            length_function = len
        )
        splits = text_splitter.split_documents(data)

        embeddings_model = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sbert-nli',
            model_kwargs={'device':'cuda'},
            encode_kwargs={'normalize_embeddings':True},
        )
        vectorstore = FAISS.from_documents(documents = splits,
                                        embedding = embeddings_model,
                                        distance_strategy = DistanceStrategy.COSINE
                                        )
        vectorstore.save_local('./db/vectorstore')
        return vectorstore

def init_bm25_retriever():
    loader = TextLoader('../data/processed_wiki_ko.txt')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 150,
        chunk_overlap = 50,
        length_function = len
    )
    splits = text_splitter.split_documents(data)

    bm25_retriever = BM25Retriever.from_documents(splits)
    with open('./db/bm25_retriever/bm25.bin', 'wb') as f:
        pickle.dump(bm25_retriever, f)

    return bm25_retriever

def retrieve_query(query):
    embeddings_model = HuggingFaceEmbeddings(
            model_name='jhgan/ko-sbert-nli',
            model_kwargs={'device':'cuda'},
            encode_kwargs={'normalize_embeddings':True},
        )
    if os.path.exists('./db/vectorstore'):
        print("Loading vectorstore")
        vectorstore = FAISS.load_local('./db/vectorstore', embeddings_model,allow_dangerous_deserialization=True)
    else:
        vectorstore = init_vectorstore()
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
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
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


if __name__ == '__main__':
    #id - 100
    query = "미적인 것은 윤리적으로 좋은 것의 상징이다.  미적인 것은  다른 모든 사람들의 동의를 요구하며 요구해야 마땅하다 .  이때   우리의 마음은 쾌락의 단순한 감각적 수용을 넘어선 순화와 고양을 의식하며 ,  다른 사람들의 가치도 그들이 지닌 판단력의   비슷한 준칙에 따라서 평가하게 된다. '다음을 주장한 사상가의 입장으로 가장 적절한 것은 ?"
    ret_docs = retrieve_query(query)
    print(ret_docs)
    print("="*50)
    print(ret_docs[0].page_content)