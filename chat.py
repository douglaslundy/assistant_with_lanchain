#pip install streamlit python-dotenv
#pip install -U langchain-openai langchain-community
#pip install faiss-gpu
#pip install faiss-cpu
#pip list


import streamlit as st
import pandas as pd
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
#from langchain_core.chains import RunnableSequence
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

loader = CSVLoader(file_path='./historia_brasil_com_temas.csv', encoding='utf-8')
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)



def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]


# retrieve_info('quem descobriu o brasil?')

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")


template = """
    Você é um Assistente virtual criado pelo Desenvolvedor Douglas Lundy, sua função será responder todas as perguntas com a melhor respostas possivel, vou lhe fornecer 
    uma base de dados, para que você utilize como sendo sua fonte de conhecimento.
    
    Siga todas as regras abaixo:
    1/ Você devera se comportar de forma educada e gentil.
    2/ Suas respostas devem ser bem similares ou até identicas as encontradas em sua base de conhecimento.
    3/ Sempre finalize a resposta de forma gentil e perguntando se pode ajudar em algo mais?
    Aqui esta a pergunta. 
    {message}
    Aqui esta sua base de conhecimentos.
    {base}
    Escreva a melhor resposta que você encontrar.
"""

prompt = PromptTemplate(
    input_variables=["message", "base"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    answers = retrieve_info(message)
    response = chain.run(message=message, base=answers)
    return print(response)


def main():
    message = input("Escreva sua pergunta. ")

    if message:
        result = generate_response(message)
        return result

if __name__ == '__main__':
    main()