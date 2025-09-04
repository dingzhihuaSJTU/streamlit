from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from openai_embedding import OpenAIEmbeddings
import os

# import sys
# import os

# sys.stderr = open(os.devnull, 'w')

# 知识库数据读取
def read_knowledge_base(file_path, state=True):
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loader = PyMuPDFLoader(file_path)
    elif file_type == 'md':
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")
    content_pages = loader.load()
    if state:
        print(f"载入后的变量类型为：{type(content_pages)}，",  f"该文件一共包含 {len(content_pages)} 页")
        md_page = content_pages[0]
        print(f"每一个元素的类型：{type(md_page)}.", 
            f"该文档的描述性数据：{md_page.metadata}", 
            f"查看该文档的内容:\n{md_page.page_content[0:][:100]}", 
            sep="\n------\n")
    return content_pages

# 多文件读取
def read_muti_knowledge(folder_path, state=True):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    texts = []
    for file_path in file_paths:
        try:
            texts.extend(read_knowledge_base(file_path, state))
        except:
            continue
    return texts

# 数据清洗
def data_clean(content_pages):
    for content_page in content_pages:
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        content_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), content_page.page_content)
        content_page.page_content = content_page.page_content.replace('•', '')
        content_page.page_content = content_page.page_content.replace(' ', '')
        content_page.page_content = content_page.page_content.replace('\n\n', '\n')
    return content_pages

# 数据去重
def data_deduplicate(content_pages):
    seen = set()
    unique_pages = []
    for content_page in content_pages:
        if content_page.page_content not in seen:
            seen.add(content_page.page_content)
            unique_pages.append(content_page)
    return unique_pages

# 数据切割
def data_split(content_pages, CHUNK_SIZE=500, OVERLAP_SIZE=100, state=True):
    # 使用递归字符文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )
    split_docs = text_splitter.split_documents(content_pages)
    if state:
        print(f"切分后的文件数量：{len(split_docs)}")
        print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
    return split_docs

# 构建知识库
def build_knowledge_base(split_docs, 
                         embeddings, 
                         persist_directory='../../data_base/vector_db/chroma', 
                         state=True):
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        # telemetry=False  # 关闭遥测
        )
    if state:
        print(f"向量库中存储的数量：{vectordb._collection.count()}")
    return vectordb

# 向量检索
def vector_search(vectordb, query, k=10, state=True):
    docs = vectordb.max_marginal_relevance_search(query, k=k)
    if state:
        for i, sim_doc in enumerate(docs):
            print(f"检索到的第{i+1}个内容: \n{sim_doc.page_content}", end="\n--------------\n")
    return docs

def creat_db(folder_path, 
             embedding, 
         CHUNK_SIZE=500, 
         OVERLAP_SIZE=100, 
         persist_directory='../data_base/vector_db/chroma', 
         query='test',  
         k=10, 
         state=True):
    # 读取知识库
    content_pages = read_muti_knowledge(folder_path, state)
    # 数据清洗
    content_pages = data_clean(content_pages)
    # 数据去重
    content_pages = data_deduplicate(content_pages)
    # 数据切割
    split_docs = data_split(content_pages, CHUNK_SIZE, OVERLAP_SIZE, state)
    # 构建知识库
    vectordb = build_knowledge_base(split_docs, embedding, persist_directory, state)
    # 向量检索
    docs = vector_search(vectordb, query, k, state=False)
    return vectordb, docs

if __name__=="__main__":
    query="什么是大模型"
    folder_path = "../data_base/knowledge_db/"
    embedding = OpenAIEmbeddings(api_key="sk-xxxxxxxxxxx")
    creat_db(folder_path, embedding=embedding, query=query)
