import streamlit as st
from openai import OpenAI
import re 
import numpy as np
import pandas as pd


template_df = pd.read_excel('assets/template_mts.xlsx') # prompt_id, msg_system_template, msg_user_retrieval_template, msg_user_score_template, trait_1~4, decription_1~4, rubric_1~4 
template_list = [item.to_dict() for i, item in template_df.iterrows()]

# set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"
# initialize chat history
if "messages" not in st.session_state:
    st.session_state.trait_messages = []
    st.session_state.trait_scores = {f'trait_{trait_idx}': '-' for trait_idx in [1,2,3,4]}
    np.random.seed(42)
    st.session_state.aggregated_scores = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], size=100, replace=True,\
                                                          p = [0.03, 0.07, 0.10, 0.15, 0.20, 0.10, 0.15, 0.10, 0.05, 0.03, 0.02])

def sub(pattern, repl, string):
    # escape backslash
    repl = str(repl).replace('\\', '\\\\')
    result = re.sub(pattern, repl, string)
    return result

def fill_msg_system_template(msg_system_template, trait):
    msg_system_template = sub('@trait', trait, msg_system_template)
    return msg_system_template

def fill_msg_user_retrieval_template(msg_user_retrieval_template, prompt, essay, trait):
    msg_user_retrieval_template = sub('@prompt', prompt, msg_user_retrieval_template)
    msg_user_retrieval_template = sub('@essay', essay, msg_user_retrieval_template)
    msg_user_retrieval_template = sub('@trait', trait, msg_user_retrieval_template)
    return msg_user_retrieval_template
    
def fill_msg_user_score_template(msg_user_score_template, trait, trait_rubric):
    msg_user_score_template = sub('@trait', trait, msg_user_score_template)
    msg_user_score_template = sub('@rubric', trait_rubric, msg_user_score_template)
    return msg_user_score_template

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

def update_history(messages: list[dict], role: str, content: str):
    messages.append({
        'role': role,
        'content': content,
    })
    return

def count_chinese_characters(text):
    """Count the number of Chinese characters in the text."""
    # chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    # return len(chinese_characters)
    return len(text)

def count_chinese_sentences(text):
    """Count the number of sentences in Chinese text."""
    # Split text by common Chinese sentence delimiters (e.g., 。！？)
    sentences = re.split(r'[。！？]', text)
    # Filter out empty sentences
    sentences = [sentence for sentence in sentences if sentence.strip()]
    return len(sentences)

# add title
st.title("Multi Trait Specialization for Zero-shot Essay Scoring 🤗")

# predefined list of essay topics with a placeholder option.
essay_topics = [x['prompt'].split('\n')[0] for x in template_list]

# sidebar for topic selection with radio buttons
st.sidebar.header("主题列表")
selected_topic = st.sidebar.radio("请选择一个主题。", essay_topics, index=None)
# render selected topic as the subheader, otherwise render a warning message that requires choosing a topic
if selected_topic:
    selected_index = essay_topics.index(selected_topic)
    template = template_list[selected_index]
    st.subheader(f"{template['prompt']}")
else:
    st.warning("请先在左侧栏中选择作文主题。")

# form to submit the input essay.
with st.form("input_essay"):
    input_essay = st.text_area(
        "作文 (最少3个句子&最少60个字):", height=150, max_chars=500,
    )
    submitted = st.form_submit_button("开始评分")

if selected_topic:
    col1, col2, col3, col4 = st.columns(4)
    # Create placeholders for the metrics
    col1, col2, col3, col4 = st.columns(4)
    placeholder1 = col1.empty()
    placeholder2 = col2.empty()
    placeholder3 = col3.empty()
    placeholder4 = col4.empty()

    # Initialize metrics with default values (e.g., '-')
    placeholder1.metric(f"**{template['trait_1']}**", f"{st.session_state.trait_scores['trait_1']}/10")
    placeholder2.metric(f"**{template['trait_2']}**", f"{st.session_state.trait_scores['trait_2']}/10")
    placeholder3.metric(f"**{template['trait_3']}**", f"{st.session_state.trait_scores['trait_3']}/10")
    placeholder4.metric(f"**{template['trait_4']}**", f"{st.session_state.trait_scores['trait_4']}/10")

    
    container = st.container()
    container.subheader('整体得分 (1-9 Band)')
    final_score_placeholder = container.empty()
    final_score_placeholder.metric(f"作文的最终分数 (1-9 Band)", "- Band", label_visibility='collapsed')

if submitted and selected_topic:
    num_sentences = count_chinese_sentences(input_essay)
    num_characters = count_chinese_characters(input_essay)

    if num_sentences < 3:
        st.warning("作文的句子数不足 3，请补充更多内容。")
    elif num_characters < 60:
        st.warning("作文的字数不足 60，请补充更多内容。")
    else:
        st.success("作文长度符合要求，开始评分...")

        for trait_idx in [1, 2, 3, 4]:
            # initiate a new conversation for each trait
            st.session_state.trait_messages = []
            msg_system = fill_msg_system_template(template['msg_system_template'], template[f'trait_{trait_idx}'])
            update_history(st.session_state.trait_messages, 'system', msg_system)
            # user message retrieval
            msg_user_retrieval = fill_msg_user_retrieval_template(template['msg_user_retrieval_template'], template['prompt'], input_essay, template[f'trait_{trait_idx}'])
            with st.chat_message("user"):
                st.write(msg_user_retrieval.split('\n')[-1])
            update_history(st.session_state.trait_messages, 'system', msg_user_retrieval)
            # assistant message retrieval
            msg_assistant_retrieval_stream = generate_assistant_message(st.session_state.trait_messages)
            with st.chat_message("assistant"):
                msg_assistant_retrieval = st.write_stream(msg_assistant_retrieval_stream)
            update_history(st.session_state.trait_messages, 'system', msg_assistant_retrieval)
            # user message score
            msg_user_score = fill_msg_user_score_template(template['msg_user_score_template'], template[f'trait_{trait_idx}'], template[f'rubric_{trait_idx}'])
            with st.chat_message("user"):
                st.write(msg_user_score)
            update_history(st.session_state.trait_messages, 'system', msg_user_score)
            # assistant message score
            msg_assistant_score_stream = generate_assistant_message(st.session_state.trait_messages)
            with st.chat_message("assistant"):
                msg_assistant_score = st.write_stream(msg_assistant_score_stream)
                score = int(re.search(r'(\d+)/10分', msg_assistant_score).groups()[0])
                st.session_state.trait_scores[f'trait_{trait_idx}'] = score

                # Update the metric placeholders dynamically
                if trait_idx == 1:
                    placeholder1.metric(f"**{template['trait_1']}**", f"{st.session_state.trait_scores['trait_1']}/10")
                elif trait_idx == 2:
                    placeholder2.metric(f"**{template['trait_2']}**", f"{st.session_state.trait_scores['trait_2']}/10")
                elif trait_idx == 3:
                    placeholder3.metric(f"**{template['trait_3']}**", f"{st.session_state.trait_scores['trait_3']}/10")
                elif trait_idx == 4:
                    placeholder4.metric(f"**{template['trait_4']}**", f"{st.session_state.trait_scores['trait_4']}/10")

            update_history(st.session_state.trait_messages, 'system', msg_assistant_score)


        input_essay_score_agg = np.mean(list(st.session_state.trait_scores.values()))
        # # round outliers
        # score_agg = np.array(st.session_state.aggregated_scores.tolist() + [input_essay_score_agg])

        # q1 = np.quantile(score_agg, 0.25) 
        # q3 = np.quantile(score_agg, 0.75) 
        # iqr = q3 - q1
        # iqr_width = 1.5
        # score_agg = np.where(score_agg < (q1-iqr*iqr_width), q1-iqr*iqr_width, score_agg)
        # score_agg = np.where(score_agg > (q3+iqr*iqr_width), q3+iqr*iqr_width, score_agg)

        # score_min = 1
        # score_max = 9
        # score_scaled_0_1 = (score_agg - score_agg.min()) / (score_agg.max() - score_agg.min())
        # score_scaled_target = score_scaled_0_1 * (score_max - score_min) + score_min

        # final_score = score_scaled_target[-1]

        final_score = (input_essay_score_agg / 10) * 8 + 1
        final_score_placeholder.metric(f"作文的最终分数 (1-9 Band)", f"{np.round(final_score).astype(int)} Band", label_visibility='collapsed')
       