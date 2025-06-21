import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=0.5, model="llama3-70b-8192", max_tokens=256)

def query_refiner(conversation, query):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the following conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}"
            ),
            ("human", "Query: {query}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "conversation": conversation,
            "query": query
        }
    )
    return response.content