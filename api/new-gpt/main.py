import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import traceback

# Load environment variables from .env file
load_dotenv()

# Fetch the entire text from Google Cloud URL as the question
def fetch_question():
    url = "https://storage.googleapis.com/fir_reccomendation/fir_text.txt"  # Replace with the actual Google Cloud URL
    response = requests.get(url)

    if response.status_code == 200:
        text = response.text
        return text
    else:
        print("Failed to fetch question. Status code:", response.status_code)

    return None

app = Flask(__name__)
CORS(app)

@app.route('/api/new-gpt', methods=['POST'])
def my_endpoint():
    try:
        # Get the question from the Google Cloud URL
        question = fetch_question()

        if not question:
            return jsonify({'error': 'Failed to fetch question from the Google Cloud URL.'}), 500

        # Get the OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not found.'}), 500

        # Determine the file path
        file_path = 'EU-AI-ACT-2.txt'

        # Load the text documents
        loader = TextLoader(file_path, encoding='utf8')
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        chat_history = []

        # Create the embeddings and Chroma index
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(texts, embeddings)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Set up the prompt template
        context = "The EU-AI-ACT is a regulatory framework governing AI development, deployment, and use in the European Union. It addresses ethics, transparency, accountability, and data protection. The goal is to ensure AI respects rights, promotes fairness, and manages risks. It includes articles on AI impact assessments, high-risk systems, human oversight, and public administration, fostering a responsible AI ecosystem."
        prompt_template = """Read the question text, understand the use case and context, and search relevant information from the EU-AI-ACT document to display recommendations based on the use case. Note that recommendations should be separated by a "%" symbol, and there should be four recommendations in total. Each reccomendation should be short and precise with article numbers and must be seperated by "%". The aim is to make suggestions that comply with the EU-AI-ACT document.

        Context:
        {context}

        Question: {question}"""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        LLM = OpenAI(temperature=0.2,model_name="gpt-3.5-turbo")
        # Set up the RetrievalQA chain
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(llm=LLM, chain_type="stuff", retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":2}), chain_type_kwargs=chain_type_kwargs, memory=memory)

        # Run the query
        answer = qa.run(question)
        print("Answer:", answer)

        response = jsonify({'answer': answer})
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/check', methods=['GET'])
def check_endpoint():
    return jsonify({'message': 'API endpoint is working.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8081)))
