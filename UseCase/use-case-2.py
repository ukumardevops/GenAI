import os
import streamlit as st
import pandas as pd

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def main():
    try:
        load_dotenv()

        st.set_page_config(
            page_icon=":books:",
            page_title="Interact with CSV Data ..."
        )

        st.title("HR - Attrition Analytics")
        st.subheader("Helps you to understand HR Employees Attritions!")

        st.markdown("""
                    This chatbot is created to demonstrate how CSV Tabular Data can be processed and analyzed by the model.
                    """)

        user_question = st.text_input(
            "Ask your questions about HR Employees Attritioning ...")

        csv_path = "./use-cases/data/hr-employees-attritions-internet.csv"

        df = pd.read_csv(csv_path)

        llm = ChatOpenAI(
            temperature=0,
            max_tokens=1000,
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model_name="gpt-4"
        )

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
        )

        agent.handle_parsing_errors = True

        answer = agent.invoke(user_question)

        st.write(answer["output"])
    except Exception as error:
        print(f"Error Occurred, Details : {error}")


if __name__ == "__main__":
    main()



# Questions to be queried:

# Give me the breakup of the number of travelers and non-travelers
# give the average tenure of people who resigned and people who did not
# is tenure in the company a good trigger for resignation? what's the degree of correlation?
# which are the attributes that contribute most of people resigning?
# are there more males or females?


#Exercise:

#    - ask the application to list out males and female employees by department
    
#    - how many female employees working in sales department?
