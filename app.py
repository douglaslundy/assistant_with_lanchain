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


# Load environment variables
load_dotenv()

# Global variables for vector store and LLM chain
db = None
chain = None

# Function to set up the vector store and embeddings
def up():
    global db
    try:
        # Load CSV data
        loader = CSVLoader(file_path='./historia_brasil_com_temas.csv', encoding='utf-8')
        documents = loader.load()

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Initialize FAISS vector store
        db = FAISS.from_documents(documents, embeddings)
        st.success("Base de dados carregada e embedders criados com sucesso.")
    except Exception as e:
        st.error(f"Erro ao carregar a base de dados e criar embedders: {e}")

# Function to retrieve information based on the query
def retrieve_info(query):
    try:
        similar_response = db.similarity_search(query, k=3)
        return [doc.page_content for doc in similar_response]
    except Exception as e:
        st.error(f"Erro ao buscar informações: {e}")
        return []

# Function to set up LLM and the prompt template
def upLLmAndTemplate():
    global chain
    try:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

        template = """
            Você é um Assistente Virtual desenvolvido pelo Douglas Lundy. Sua função é responder perguntas com base em uma base de conhecimentos específica. Siga as diretrizes abaixo para garantir a melhor experiência ao usuário:

        Cordialidade e Gentileza: Sempre se comunique de forma educada e amigável.
        Fidelidade à Base de Conhecimento: Suas respostas devem estar exclusivamente alinhadas com as informações contidas na base fornecida. Utilize somente a base de conhecimento {base}.
        Encerramento Positivo: Termine cada resposta com uma expressão gentil e ofereça ajuda adicional, como: "Posso ajudar em algo mais?".
        Proibição de Informações Externas: Não responda perguntas sobre assuntos que não estão na base {base}. Se o conteúdo não for encontrado na base, responda exatamente com: "Eu ainda não aprendi sobre esse assunto."
        Sem Consultas Externas: Não busque informações fora da base de conhecimento fornecida e não faça suposições ou especulações sobre qualquer tópico.

        Pergunta do Usuário:
        {message}

        Base de Conhecimentos:
        {base}
        """

        prompt = PromptTemplate(
            input_variables=["message", "base"],
            template=template
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        st.success("LLM e template configurados com sucesso.")
    except Exception as e:
        st.error(f"Erro ao configurar LLM e template: {e}")

# Function to generate a response from the model
def generate_response(message):
    try:
        answers = retrieve_info(message)
        response = chain.run(message=message, base=answers)
        return response
    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")
        return "Erro ao gerar resposta."

# Main function to integrate everything
def main():
    st.title("Assistente Virtual de História do Brasil")
    st.write("Pergunte algo sobre a história do Brasil.")

    # Setup the embeddings and LLM chain
    up()
    upLLmAndTemplate()

    # User input for question
    message = st.text_input("Escreva sua pergunta:")

    if message:
        result = generate_response(message)
        st.write(result)

# Entry point of the script
if __name__ == '__main__':
    main()