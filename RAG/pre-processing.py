from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

import os


def get_pdf_text(pdf_document):
    text = ""

    pdf_reader = PdfReader(pdf_document)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def create_documents(pdf_files):
    documents = []

    for file in pdf_files:
        chunks = get_pdf_text(file)

        documents.append(
            Document(
                page_content=chunks,
                metadata={
                    "source": file,
                    "type": "PDF",
                    "owner": "ukumar"
                }
            )
        )

    return documents


def create_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    return embeddings


def push_documents_to_pinecone(index_name, embeddings, documents):
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    vector_store.add_documents(documents)


def main():
    try:
        load_dotenv()

        index_name = "uttam-resumes"
        directory_path = "./resume"
        files = os.listdir(directory_path)
        pdf_files = []

        for file in files:
            pdf_file = directory_path + "/" + file
            pdf_files.append(pdf_file)

            print(f"Processing Required ... PDF Document {pdf_file} ...")

        documents = create_documents(pdf_files)
        embeddings = create_embeddings()

        push_documents_to_pinecone(index_name, embeddings, documents)

        print("Vector Embeddings are successfully Processed into the Vector Database ...")
    except Exception as error:
        print(f"Error Occurred, Details : {error}")


if __name__ == "__main__":
    main()