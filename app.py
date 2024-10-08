from flask import Flask, render_template,request, jsonify
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from agent_helper import create_vector_store
from langchain.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
load_dotenv()

app= Flask(__name__)

local_llm= "neural-chat-7b-v3-1.Q4_K_M.gguf"

llm= CTransformers(model= local_llm, model_type='mistral',lib='avx2')
print("LLm initialized")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])


load_vector_store = Chroma(persist_directory="Vectorstores/documents", embedding_function=embeddings)

retriever = load_vector_store.as_retriever(search_kwargs={"k":1})



@app.route('/')
def index():
    return render_template('index.html')


@app.route("/get_response",methods=['POST'])
def get_response():
    query= request.form.get('query')
    chain_type_kwargs={"prompt":prompt}
    qa= RetrievalQA.from_chain_type(llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True)
    response= qa(query)
    answer=response['result']
    print(response)
    source_document= response['source_documents'][0].page_content
    
    doc= response['source_documents'][0].metadata['source']
    response_data = {"answer": answer, "source_document": source_document, "doc": doc}
    
    return jsonify(response_data)

if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)