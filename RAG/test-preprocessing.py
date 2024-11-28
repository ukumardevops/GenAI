from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


def create_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    return embeddings


def search_similar_documents(query, no_of_documents, index_name, embeddings):
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    similar_documents = vector_store.similarity_search(
        query, k=no_of_documents)

    return similar_documents


def main():
    try:
        load_dotenv()

        index_name = "uttam-resumes"
        query = """
            Experienced Candidates with Embedded Systems
            
            Requirements:
            
            Bachelors Degree in Computer Science
            At least Five years of Working Experience in Embedded Systems
            Understanding of Computer Architecture, Programming Languages and Interfacing Technologies 
        """

        embeddings = create_embeddings()
        no_of_documents = 2

        relevant_documents = search_similar_documents(
            query, no_of_documents, index_name, embeddings)

        for doc_index in range(len(relevant_documents)):
            document = relevant_documents[doc_index]

            print(document.metadata["source"])
            print(document.page_content)
            print("\n")
    except Exception as error:
        print(f"Error Occurred, Details : {error}")


if __name__ == "__main__":
    main()