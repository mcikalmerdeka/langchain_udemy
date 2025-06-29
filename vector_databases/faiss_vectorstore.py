import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI 
from langchain import hub

load_dotenv("../.env")


if __name__ == "__main__":
    print("Starting the application...")

    # Load the PDF file
    pdf_path = "./vector_databases/ReAct_paper.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separators=["\n\n", "\n", " ", ""])
    docs = text_splitter.split_documents(documents=documents)

    # Embed the chunks and store them in a FAISS vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings) # Note that the vectorstore will be stored in the RAM of our local machine
    vectorstore.save_local("faiss_index_react") # Persist the vectorstore to the local machine (if this is not done, the vectorstore will be lost when the program is closed)

    # Load the vectorstore from the local machine
    new_vectorstore = FAISS.load_local(
        "faiss_index_react",
        embeddings,
        allow_dangerous_deserialization=True # This is a security risk, but we are using it here because we are loading the vectorstore from the local machine
    )

    # Create the prompt template
    llm = ChatOpenAI(model="gpt-4.1")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=retrieval_qa_chat_prompt
    )

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    # Query the vectorstore about ReAct paper
    result = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(result["answer"])