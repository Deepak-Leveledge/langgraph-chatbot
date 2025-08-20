import streamlit as st
from chatbotbackend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage

user_input =st.chat_input("Type your message here...")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# message_history= []


CONFIG = {'configurable':{'thread_id':"thread_1"}}

for message in  st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])



if user_input:

    ## Add the messge to the message history

    st.session_state["message_history"].append({"role": "user", "content": user_input})


    with st.chat_message("user"):
        st.text(user_input)

    

    response =chatbot.invoke({
        "messages": [
            HumanMessage(content=user_input)
        ]
    },config=CONFIG)

    ai_message= response['messages'][-1].content



    
    ## Add the messge to the message history

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})




  
    with st.chat_message("assistant"):
        st.text(ai_message)





   