import streamlit as st
from openai import OpenAI
import numpy as np
import pandas as pd


# set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"
# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def generate_assistant_message(messages: list[dict[str, str]]):
    """
        get LLM assistant's msg from OpenAI API. 
        set generation hyperparameters here.
    """
    stream = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
        stream=True,
    )
    return stream

# add title
st.title("基于多特征专业化的作文评分系统🤗")

# predefined list of essay topics with a placeholder option.
essay_topics = [
    'library cencorship',
    'copyright issues of AI-generated artworks.',
    'the impact of movie on people\'s daily life.'
]

# sidebar for topic selection with radio buttons
st.sidebar.header("Essay Topic Selection")
selected_topic = st.sidebar.radio("Select an Essay Topic", essay_topics, index=None)
# render selected topic as the subheader, otherwise render a warning message that requires choosing a topic
if not selected_topic:
    st.warning("Please select a topic from the sidebar.")
else:
    st.subheader(f"{selected_topic}")

# form to submit the input essay.
with st.form("input_essay"):
    input_essay = st.text_area(
        "Enter text:",
    )
    submitted = st.form_submit_button("Submit")

if submitted:

    with st.chat_message("user"):
        user_msg = f"assian a score (from 0 to 10) to the following [Input Essay]. [Essay Topic]: {selected_topic}  [Input Essay]: {input_essay}"
        st.write(user_msg)
    # save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_msg,
    })

    with st.chat_message("assistant"):
        stream = generate_assistant_message(st.session_state.messages)
        response = st.write_stream(stream)