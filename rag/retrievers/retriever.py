from dotenv import load_dotenv
import os
from typing import Any, Dict, List
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

def retrieve(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model=os.environ["OPENAI_EMBEDDINGS_MODEL"])
    docsearch = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings, pinecone_api_key=os.environ["PINECONE_API_KEY"])
    chat = ChatOpenAI(model_name="gpt-4o-mini", verbose=True, temperature=0)

    # Define the rephrase prompt explicitly
    rephrase_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question."
        ),
        HumanMessagePromptTemplate.from_template(
            "Chat History:\n{chat_history}\n\nFollow Up Input: {input}\n\nStandalone question:"
        ),
    ])

    # Define the retrieval QA chat prompt explicitly
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an AI assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
        ),
        HumanMessagePromptTemplate.from_template(
            "Question: {input}\n\nContext: {context}\n\nAnswer:"
        ),
    ])

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result["answer"]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    query = "Que restaurantes servem pizza no rio de janeiro?"
    result = retrieve(query)
    print(result["answer"])