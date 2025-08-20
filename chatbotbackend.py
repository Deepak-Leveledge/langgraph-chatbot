from langgraph.graph import StateGraph , START, END 
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os


load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API"))


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    message  = state["messages"]
    response = model.invoke(message)
    return {"messages":[response]}



Checkpointer = InMemorySaver()


graph= StateGraph(ChatState)

graph.add_node("chat_node",chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


chatbot = graph.compile(
    checkpointer=Checkpointer)

# CONFIG= {"configurable":{"thread_id":"thread-1"}}

# response= chatbot.invoke({
#     "messages":[HumanMessage(content="What is the recepie of maggi")]},
#     config=CONFIG
# )

# print(chatbot.get_state(config=CONFIG))

# for message_chunk, metadata in  chatbot.stream(
#     {"messages": [HumanMessage(content="What is the recepie of maggi")]},
#     config={"configurable": {"thread_id": "thread-1"}},
#     stream_mode="messages"
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ",flush=True)

