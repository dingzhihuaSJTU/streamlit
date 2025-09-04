from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import asyncio

# import sys
# import os

# sys.stderr = open(os.devnull, 'w')

def get_llm(temperature=0, 
            openai_api_key="sk-xxxxxxxxxxx", 
            base_url="https://api.siliconflow.cn/v1", 
            model_name = "Qwen/Qwen2.5-7B-Instruct", 
            streaming=True):
    llm = ChatOpenAI(temperature=temperature, 
                     openai_api_key=openai_api_key, 
                     base_url=base_url, 
                     model_name=model_name, 
                     streaming=streaming)

    return llm

async def get_response(message):
    llm = get_llm()
    chunks = []
    async for chunk in llm.astream(message):
        chunks.append(chunk)
        print(chunk.content, end="", flush=True)
    print('\n')
    return chunks

def get_message(template, human_template, text):
    chat_prompt = ChatPromptTemplate([
        ("system", template),
        ("human", human_template),
    ])

    messages  = chat_prompt.invoke(text)
    return messages

def main():
    template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
    human_template = "{text}"
    text_str = "我带着比身体重的行李，\
                游入尼罗河底，\
                经过几道闪电 看到一堆光圈，\
                不确定是不是这里。\
                "
    text = {"input_language": "中文", "output_language": "英文", "text": text_str}
    messages = get_message(template, human_template, text)
    chunks = asyncio.run(get_response(messages))
    # print(chunks)

if __name__=="__main__":
    main()