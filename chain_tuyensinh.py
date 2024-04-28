from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from vector_store import retriever

llm = ChatOpenAI(temperature=0.9, model_kwargs={"top_p":0.95}, max_tokens=1024)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = hub.pull("bagumeow/qa-tuyensinh")
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

tuyensinh_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# conversational_rag_chain = RunnableWithMessageHistory(
#     tuyensinh_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )
# from uuid import uuid4

# if __name__ == "__main__":
#     output = {}
#     curr_key = None
#     session_id = str(uuid4())
#     input_question = None
#     while input_question != "exit":
#         input_question = input("Mời bạn nhập câu hỏi: ")

#         for chunk in conversational_rag_chain.stream({"input": input_question},
#             config={
#                 "configurable": {"session_id": session_id},
#             }, 
#         ):
#             for key in chunk:
#                 if key =="answer":
#                     if key not in output:
#                         output[key] = chunk[key]
#                     else:
#                         output[key] += chunk[key]
#                     if key != curr_key:
#                         print(f"{key}: {chunk[key]}", end="", flush=True)
#                     else:
#                         print(chunk[key], end="", flush=True)
#                     curr_key = key
#         print()
#         print("-"*100)
