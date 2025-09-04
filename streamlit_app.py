import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
sys.path.append("../") # 将父目录放入系统路径中
from openai_embedding import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chat_llm import get_llm
from knowledge_chroma import creat_db
import json
from datetime import datetime
import glob

# 检测知识库文件夹和向量库目录的最新修改时间
def get_latest_mtime(path):
    mtime = os.path.getmtime(path)
    return mtime

# 加载数据库
def load_chroma(filepath="./database/knowledge/", 
                persist_directory='./database/chroma', 
                embedding=OpenAIEmbeddings(api_key=st.session_state.get("openai_api_key", "")), 
                state=True):
    # 如果persist_directory路径下没有文件，则调用creat_db生成数据库
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        print("向量数据库不存在，正在创建中...")
        vectordb, _ = creat_db(
            folder_path=filepath, 
            embedding=embedding,
            CHUNK_SIZE=500,
            OVERLAP_SIZE=100,
            persist_directory=persist_directory,
            query='test',
            k=10,
            state=False
        )
    else:
        embedding_function = embedding
        # 向量数据库持久化路径
        vectordb = Chroma(
            persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
            embedding_function=embedding_function
        )
    if state:
        file_paths = []
        for root, dirs, files in os.walk(filepath):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
        if file_paths==[]:
            print("Warning: 知识库中没有文件")
        else:
            print("\n知识来源: ")
            for i in file_paths:
                print(f" - {i.replace('./database/knowledge/', '')}")
            print(f"\n向量库中存储的数量：{vectordb._collection.count()}\n")
    return vectordb

def get_retriever(llm):
    filepath = "./database/knowledge/"  # 知识库文件夹路径
    persist_directory = './database/chroma/'  # 向量数据库持久化路径
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    # 加载数据库
    kb_mtime = get_latest_mtime(filepath)
    chroma_mtime = get_latest_mtime(persist_directory)
    # 如果知识库有更新，则删除旧的向量库目录
    # print(chroma_mtime - kb_mtime)
    if kb_mtime > chroma_mtime:
        import shutil
        print("知识库有更新，正在删除旧的向量库...")
        shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
    try:

        vectordb = load_chroma(filepath,
                            persist_directory, 
                            embedding=OpenAIEmbeddings(api_key=st.session_state.get("openai_api_key", "")), 
                            state=True)
    except Exception as e:
        vectordb = load_chroma(filepath,
                            persist_directory, 
                            embedding=OpenAIEmbeddings(api_key=st.session_state.get("openai_api_key", "")), 
                            state=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    # 压缩问题的系统 prompt
    condense_question_system_template = (
        "请根据聊天记录完善用户最新的问题，"
        "如果用户最新的问题不需要完善则返回用户的问题。"
        )
    # 构造 压缩问题的 prompt template
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
    # 构造检索文档的链
    # RunnableBranch 会根据条件选择要运行的分支
    retrieve_docs = RunnableBranch(
        # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        # 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    return retrieve_docs

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain(model_name="Qwen/Qwen2.5-7B-Instruct"):
    temperature = st.session_state.get("temperature", 0)
    llm = get_llm(
        temperature=temperature,
        openai_api_key=st.session_state.get("openai_api_key", ""),
        base_url="https://api.siliconflow.cn/v1",
        model_name=model_name,
        streaming=True
    )
    retrieve_docs = get_retriever(llm)

    system_prompt = (
        """你是一个问答任务的助手。 
        你叫小丁同学。
        使用以下上下文来回答最后的问题。
        如果你不知道答案，就说你不知道，不要试图编造答案。
        尽量使答案简明扼要，但是又要绝对详细，且保证正确性。
        {context}
        """
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain

def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

# 聊天记录文件名统一管理
CHAT_HISTORY_DIR = "./chat_history/"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

def get_current_chat_filename():
    # 判断是否已有聊天内容
    if "chat_file" not in st.session_state:
        if st.session_state.messages and len(st.session_state.messages) > 0 and st.session_state.messages[0][1].strip():
            first_msg = st.session_state.messages[0][1].strip().replace('\n', ' ').replace(' ', '_')[:20]
            fname = f"{first_msg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            fname = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        st.session_state.chat_file = os.path.join(CHAT_HISTORY_DIR, fname)
    return st.session_state.chat_file

def save_chat_history(messages):
    filename = get_current_chat_filename()
    # 修复新建聊天时 chat_file 为 None 的问题
    if filename is None:
        # 重新生成文件名
        if messages and len(messages) > 0 and messages[0][1].strip():
            first_msg = messages[0][1].strip().replace('\n', ' ').replace(' ', '_')[:20]
            fname = f"{first_msg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            fname = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename = os.path.join(CHAT_HISTORY_DIR, fname)
        st.session_state.chat_file = filename
    # 如果没有消息，且文件存在则删除
    if not messages or len(messages) == 0:
        if filename and os.path.exists(filename):
            os.remove(filename)
        return None
    # 检查是否需要重命名
    if messages and len(messages) > 0 and messages[0][1].strip():
        first_msg = messages[0][1].strip().replace('\n', ' ').replace(' ', '_')[:20]
        expected_name = f"{first_msg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        expected_path = os.path.join(CHAT_HISTORY_DIR, expected_name)
        if os.path.basename(filename).startswith('chat_'):
            # 仅在首次有内容时重命名
            if os.path.exists(filename):
                os.rename(filename, expected_path)
                st.session_state.chat_file = expected_path
                filename = expected_path
            else:
                # 如果原文件不存在，直接用新文件名保存
                st.session_state.chat_file = expected_path
                filename = expected_path
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    return filename

def list_chat_history_files():
    files = glob.glob(os.path.join(CHAT_HISTORY_DIR, "*.json"))
    files.sort(reverse=True)
    return files

def load_chat_history(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)



# Streamlit 应用程序界面
def main():
    st.markdown('### 🔗 小丁同学帮你办')

    # 侧边栏功能和历史聊天区集成
    with st.sidebar:
        st.markdown("## 功能区")
        st.markdown("---")
        st.markdown("#### API密钥与模型参数")
        openai_api_key = st.text_input("请输入 OpenAI API Key", value=st.session_state.get("openai_api_key", ""), type="password")
        st.session_state.openai_api_key = openai_api_key
        temperature = st.slider(
            label="模型风格",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            format="%.2f"
        )
        st.session_state.temperature = temperature
        st.write(f"当前模型属于：{'理科生' if temperature<0.5 else ('文科生' if temperature>0.5 else '文理兼备')}")

        # 1. 模型选择
        model_name_dict = {
            1: "Qwen/Qwen2.5-7B-Instruct",
            2: "Pro/Qwen/Qwen2.5-7B-Instruct",
            3: "Pro/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            4: "Qwen/Qwen3-30B-A3B-Instruct-2507",
            5: "deepseek-ai/DeepSeek-V3.1",
            6: "Pro/deepseek-ai/DeepSeek-V3.1"
        }
        model_desc_dict = {
            1: "默认模型，免费，反应快，能力尚可",
            2: "1模型的pro版，反应理论上快一些，1模型卡顿可以切换为2",
            3: "deepseek小模型，有思考，或许会更准确，但是反应速度会变慢",
            4: "Qwen大模型，高质量输出可选",
            5: "deepseek大模型，能力理论上比4强",
            6: "5模型的pro版，反应更快，能力不变，慎用（超级贵）"
        }
        model_options = [f"{i}: {model_name_dict[i]}" for i in model_name_dict]
        selected_model_idx = st.selectbox("选择模型：", options=list(model_name_dict.keys()), format_func=lambda x: f"{model_name_dict[x]}")
        st.session_state.selected_model = model_name_dict[selected_model_idx]
        st.info(f"模型描述：{model_desc_dict[selected_model_idx]}")

        # 2. PDF上传
        st.markdown("---")
        st.markdown("### 上传PDF文件")
        uploaded_file = st.file_uploader("选择PDF文件", type=["pdf"])
        pdf_save_path = "./database/knowledge/"
        if not os.path.exists(pdf_save_path):
            os.makedirs(pdf_save_path, exist_ok=True)
        if uploaded_file is not None:
            if uploaded_file.type != "application/pdf":
                st.error("仅支持PDF文件！")
            else:
                save_path = os.path.join(pdf_save_path, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"文件 {uploaded_file.name} 已保存到知识库目录！")

        # 3. 生成知识库按钮
        if st.button("生成知识库"):
            pdf_files = [os.path.join(pdf_save_path, f) for f in os.listdir(pdf_save_path) if f.endswith('.pdf')]
            if not pdf_files:
                st.warning("知识库目录下没有PDF文件，无法生成知识库！")
            else:
                st.info("正在生成知识库，请稍候...")
                for pdf_file in pdf_files:
                    creat_db(
                        folder_path=pdf_save_path,
                        CHUNK_SIZE=500,
                        OVERLAP_SIZE=100,
                        embedding=OpenAIEmbeddings(api_key=st.session_state.get("openai_api_key", "")),
                        persist_directory='./database/chroma',
                        query='test',
                        k=10,
                        state=False
                    )
                st.success("知识库已生成！")

        # 4. 显示知识库中的PDF文件列表
        st.markdown("---")
        st.markdown("### 当前知识库PDF文件列表")
        pdf_files_list = [f for f in os.listdir(pdf_save_path) if f.endswith('.pdf')]
        if pdf_files_list:
            for pdf_file in pdf_files_list:
                st.write(f"- {pdf_file}")
        else:
            st.info("知识库目录下暂无PDF文件。")

        # 5. 历史聊天记录区
        st.markdown("---")
        st.markdown('## 历史聊天记录')
        chat_files = list_chat_history_files()
        if chat_files:
            for chat_file in chat_files:
                chat_title = os.path.basename(chat_file).replace('.json','')
                if st.button(chat_title, key=f"load_{chat_title}"):
                    if st.session_state.get('chat_file') != chat_file:
                        st.session_state.messages = load_chat_history(chat_file)
                        st.session_state.chat_file = chat_file
        else:
            st.info("暂无历史聊天记录。")

        

    # 用于跟踪对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state or st.session_state.get("qa_chain_model_name") != st.session_state.selected_model:
        st.session_state.qa_history_chain = get_qa_history_chain(model_name=st.session_state.selected_model)
        st.session_state.qa_chain_model_name = st.session_state.selected_model

    # 主页面只保留聊天区相关内容

    if st.button("新建聊天", key="new_chat"):
        st.session_state.messages = []
        st.session_state.chat_file = None
    
    chat_container = st.container()
    with chat_container:
        st.info("我是你的知识库助手，有什么可以帮你？")
        # 展示历史消息，AI消息增加复制按钮
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message[0]):
                if message[0] == "ai":
                    st.write(message[1])
                    # 复制按钮
                    if st.button("📋", key=f"copy_{idx}"):
                        # st.session_state["copy_text"] = message[1]
                        st.code(message[1], language=None)
                else:
                    st.write(message[1])
    prompt = st.chat_input("Say something")
    
    if prompt:
        st.session_state.messages.append(("human", prompt))
        with st.chat_message("human"):
            st.write(prompt)
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        with st.chat_message("ai"):
            output = st.write_stream(answer)
            if st.button("📋", key="new_copy"):
                # st.session_state["copy_text"] = message[1]
                st.code(output, language=None)
        st.session_state.messages.append(("ai", output))
        save_chat_history(st.session_state.messages)
    # with col_history:

if __name__ == "__main__":
    
    main()



