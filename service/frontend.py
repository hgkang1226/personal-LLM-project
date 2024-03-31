import streamlit as st
from streamlit_chat import message
import pandas as pd
import requests
import json
from datetime import datetime


# streamlit run frontend.py

class llama_web():
    def __init__(self):
        self.URL = "http://127.0.0.1:8000/chat"
        self.count = 0
        st.set_page_config(
            page_title="Streamlit Chat -Demo",
            page_icon="robot"
        )

    def get_answer(self, message):
        param = {'user_message': message}
        resp = requests.post(self.URL, json=param)
        output = json.loads(resp.content)
        output = output['message']
        return output
        # return message

    def update_log(self, user_message, bot_message):
        if 'chat_log' not in st.session_state:
            st.session_state.chat_log = {"user_message": [], "bot_message": [], "timestamp": []}
        st.session_state.chat_log['user_message'].append(user_message)
        st.session_state.chat_log['bot_message'].append(bot_message)
        st.session_state.chat_log['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        return st.session_state.chat_log

    def chat(self):
        st.title("Personal LLM Project")
        st.header("Chatting with LLaMA")
        st.subheader("Please type a message only in English to chat with LLaMA.")
        st.caption("Hyun-Gook Kang")

        # 채팅 로그를 스크롤 가능한 컨테이너에 출력
        chat_container = st.container()
        with chat_container:
            if 'chat_log' in st.session_state:
                chat_log = st.session_state.chat_log
                bot_messages = chat_log['bot_message']
                user_messages = chat_log['user_message']
                timestamps = chat_log['timestamp']
                for idx, (bot, user, timestamp) in enumerate(zip(bot_messages, user_messages, timestamps)):
                    message(user, key=str(idx), is_user=True, avatar_style="big-smile")
                    st.caption(timestamp)
                    message(bot, key=f"{idx}_bot", avatar_style="bottts")

        # 텍스트 입력창을 화면 하단에 고정
        user_message = st.text_input("type a message..", key="user_message", on_change=self.handle_user_input)


    def handle_user_input(self):
        user_message = st.session_state.user_message
        if user_message:
            output = self.get_answer(user_message)
            chat_log = self.update_log(user_message, output)
            st.session_state.user_message = ""  # 입력창 초기화

if __name__ == '__main__':
    web = llama_web()
    web.chat()
    