from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from chain_tuyensinh import tuyensinh_chain
from vector_store import retriever
from uuid import uuid4
import json
import os

class ChatBotTuyenSinh:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.9, model_kwargs={"top_p":0.95}, max_tokens=1024)
        self.bot_tuyensinh = RunnableWithMessageHistory(
                tuyensinh_chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        self.store = {}

    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def chat(self):
        if not os.path.exists("history_chat"):
            os.makedirs("history_chat")
        output = {}
        curr_key = None
        session_id = str(uuid4())
        print(f"Session_id: {session_id}")
        input_question = None 
        while input_question != "exit":
            input_question = input("Mời bạn nhập câu hỏi: ")

            for chunk in self.bot_tuyensinh.stream({"input": input_question},
                config={
                    "configurable": {"session_id": session_id},
                }, 
            ):
                for key in chunk:
                    if key =="answer":
                        if key not in output:
                            output[key] = chunk[key]
                        else:
                            output[key] += chunk[key]
                        if key != curr_key:
                            print(f"{key}: {chunk[key]}", end="", flush=True)
                        else:
                            print(chunk[key], end="", flush=True)
                        curr_key = key
            print()
            print("-"*100)
            self.save_session_history(session_id= session_id)

    
    def save_session_history(self,session_id:str):
        store =  {session_id: str(self.get_session_history(session_id).messages)}
        with open(f"history_chat/{session_id}.json", "w",encoding="utf-8") as history:
            json.dump(store, history, ensure_ascii=False, indent=4)
        
    
if __name__ == "__main__":
    bot = ChatBotTuyenSinh()
    bot.chat()
    
    # print(bot.get_session_history(session_id))
        

    
    