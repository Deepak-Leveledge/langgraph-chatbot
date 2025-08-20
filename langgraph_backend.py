from langgraph.graph import StateGraph , START, END 
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import sqlite3


load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API"))


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    message  = state["messages"]
    response = model.invoke(message)
    return {"messages":[response]}


conn = sqlite3.connect(database="chatbot.db",check_same_thread=False)

Checkpointer = SqliteSaver(conn=conn)


graph= StateGraph(ChatState)

graph.add_node("chat_node",chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


chatbot = graph.compile(
    checkpointer=Checkpointer)


def retrive_all_threads():
    """
    Retrieve all the threads from the Checkpointer.

    Returns:
        List[str]: List of all the threads.
    """
    all_threads = set()
    for checkpoint in Checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)




# CONFIG = {"configurable":{"thread_id":"thread-1"}}

# result = chatbot.invoke(
#     {"messages":[HumanMessage(content="what is my name")]},
#     config=CONFIG
# )

# print(result)