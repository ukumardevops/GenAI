from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
import os


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


def get_summary_from_llm(current_document):
    load_dotenv()

    openai_api_key = os.environ["OPENAI_API_KEY"]

    llm = ChatOpenAI(
        temperature=0,
        max_tokens=2000,
        openai_api_key=openai_api_key
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_document])

    return summary


def main():
    try:
        load_dotenv()
        
        index_name = "uttam-resumes"

        st.set_page_config(page_title="Resume Screening Assistant")
        st.title("Resume Screening AI Assistant")
        st.subheader(
            """
            This AI Assistant would help you to screen available resumes that have been indexed.
            """
        )

        job_description = st.text_area(
            "Please enter or paste your Job Description ...",
            key="1",
            height=200
        )

        document_count = st.text_input(
            "No. Of Matching Resume(s) to be shown ...")

        submit = st.button("Analyze")

        if submit:
            embeddings = create_embeddings()
            relevant_documents = search_similar_documents(
                job_description,
                int(document_count),
                index_name,
                embeddings
            )

            for document_index in range(len(relevant_documents)):
                st.subheader(f"{str(document_index+1)} Document ...")

                file_name = "** FILE **" + \
                    relevant_documents[document_index].metadata["source"]

                st.write(file_name)

                with st.expander("Show me Summary ..."):
                    summary = get_summary_from_llm(
                        relevant_documents[document_index])

                    st.write("**** SUMMARY ****" + summary)
    except Exception as error:
        print(f"Error Occurred, Details : {error}")


if __name__ == "__main__":
    main()