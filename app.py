# -*- coding: utf-8 -*-
import random
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import openai
# from playsound import playsound
from gtts import gTTS
from PyPDF2 import PdfReader
from utils import text_to_docs
from langchain import PromptTemplate, LLMChain
#import os
#from io import StringIO
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.chains.summarize import load_summarize_chain
#import os
#import pyaudio
#import wave
#import langchain
#from langchain.document_loaders import UnstructuredPDFLoader
#from io import BytesIO
# import streamlit.components.v1 as components
#from st_custom_components import st_audiorec, text_to_docs


#import sounddevice as sd
#from scipy.io.wavfile import write
from usellm import Message, Options, UseLLM


st.title("Suspicious Activity Reporting")
st.subheader('Evidence Processor')


model_name = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_data
def usellm(prompt):

    service = UseLLM(service_url="https://usellm.org/api/llm")
    messages = [
      Message(role="system", content="You are a fraud analyst, who is an expert at finding out suspicious activities"),
      Message(role="user", content=f"{prompt}"),
      ]
    options = Options(messages=messages)
    response = service.chat(options)
    return response.content


@st.cache_resource
def embed(model_name):
    hf_embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    return hf_embeddings


hf_embeddings = embed(model_name) 

# File Upload
file = st.file_uploader("Upload a file")

st.write(file)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 100,
    length_function = len,
    separators=["\n\n", "\n", " ", ""]
)
#text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=0)
#texts = ''
@st.cache_data
def embedding_store(file):
    # save file
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    #st.write(text)
    texts =  text_splitter.split_text(text)
    docs = text_to_docs(texts)
    #st.write(texts)
    docsearch = FAISS.from_documents(docs, hf_embeddings)
    return docs, docsearch

# Submit Button
if st.button("Submit"):
    if file is not None:
        # File handling logic
        st.write("File Uploaded...")
        _, docsearch = embedding_store(file)
        queries ="Please provide the following information regarding the fraud case based on the uploaded file: Victim's Name,Existence of any reported suspect\
        List of Merchant names, How the bank was notified, Date of bank notification, Type of Fraud, Date of the fraud occurrence\
        Whether the disputed amount exceeded 5000 USD, Types of cards involved, Whether a police report was filed\
        Assessment of suspicious activity based on the evidence, if any"
        contexts = docsearch.similarity_search(queries, k=1) 
        prompts = f" Give concise answer to the below questions as truthfully as possible as per given context only,\n\n\
                1. What is the Victim's Name?\n\
                2. Has any suspect been reported?\n\
                3. List the Merchant name\n\
                4. How was the bank notified?\n\
                5. When was the bank notified?\n\
                6. What is the Fraud Type?\n\
                7. When did the fraud occur?\n\
                8. Was the disputed amount greater than 5000 USD?\n\
                9. What type of cards are involved?\n\
                10. Was the police report filed?\n\
                11. Based on the evidence is this a suspicious activity?\n\
              Context: {contexts}\n\
              Response (in readable bullet points): "
              

        response = usellm(prompts)
        st.write(response)

#st.write("Uploaded File Contents:")
if file is not None:
    docs, docsearch = embedding_store(file)

# Text Input
st.subheader("Ask Questions")
query = st.text_input('your queries will go here...')

def LLM_Response():
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run({"query":query, "context":context})
    return response


        
if query:
    # Text input handling logic
    #st.write("Text Input:")
    #st.write(text_input)
    context_1 = docsearch.similarity_search(query, k=5)

    if query.lower() == "what is the victim's name?":
        prompt_1 = f'''Perform Name Enitity Recognition to identify the Customer name as accurately as possible, given the context. The Customer can also be referenced as the Victim or the person with whom the Fraud has taken place.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

        
    elif query.lower() == "what is the suspect's name?":
        prompt_1 = f'''Perform Name Enitity Recognition to identify the Suspect name as accurately as possible, given the context. Suspect is the Person who has committed the fraud with the Customer. Respond saying "The Suspect Name is not Present" if there is no suspect in the given context.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

        
    elif query.lower() == "list the merchant name":
        prompt_1 = f'''Perform Name Enitity Recognition to identify all the Merchant Organizations as accurately as possible, given the context. A merchant is a type of business or organization that accepts payments from the customer account. Give a relevant and concise response.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

        
    elif query.lower() == "how was the bank notified?":
        prompt_1 = f''' You need to act as a Financial analyst to identify how was the bank notified of the Supicious or Fraud event with in the given context. The means of communication can be a call, an email or in person. Give a relevant and concise response.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

        
    elif query.lower() == "when was the bank notified?":
        prompt_1 = f''' You need to act as a Financial analyst to identify the when the bank was notified of the Fraud i.e., the disputed date. Given the context, provide a relevant and concise response.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

        
    elif query.lower() == "what type of fraud is taking place?":
        prompt_1 = f''' You need to act as a Financial analyst to identify the type of fraud or suspicious activity has taken place amd summarize it, within the given context. Also mention the exact fraud code. Give a relevant and concise response.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

        
    elif query.lower() == "when did the fraud occur?":
        prompt_1 = f''' You need to act as a Financial analyst to identify the when the did the fraud occur i.e., the Transaction Date. Given the context, provide a relevant and concise response.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

            
    elif query.lower() == "was the disputed amount greater than 5000 usd?":
        prompt_1 = f''' You need to act as a Financial analyst to identify the disputed amount and perform a mathematical calculation to check if the disputed amount is greater than 5000 or no, given the context. Give a relevant and concise response.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

        
    elif query.lower() == "what type of cards are involved?":
        prompt_1 = f''' You need to act as a Financial analyst to identify the type of card and card's brand involved, given the context. On a higher level the card can be a Credit or Debit Card. VISA, MasterCard or American Express, Citi Group, etc. are the different brands with respect to a Credit card or Debit Card . Give a relevant and concise response.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

        
    elif query.lower() == "was the police report filed?":
        prompt_1 = f''' You need to act as a Financial analyst to identify if the police was reported of the Fraud activity, given the context. Give a relevant and concise response.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

            
    elif query.lower() == "Is this a valid SAR case?":
        prompt_1 = f''' You need to act as a Financial analyst to check if this is a SAR or not, given the following context, if the transaction amount is less than 5000 USD we cannot categorize this as SAR (Suspicious activity Report).Give a relevant and concise response. \n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\
                      Response: '''

        
    else:
        prompt_1 = f'''Act as a financial analyst and give concise answer to below Question as truthfully as possible, with given Context.\n\n\
                      Question: {query}\n\
                      Context: {context_1}\n\                      
                      Response: '''


    #prompt = PromptTemplate(template=prompt, input_variables=["query", "context"])
    response = usellm(prompt_1) #LLM_Response()
    st.write(response)



# Footer
st.markdown(
    """
    <style>
      MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """
    , unsafe_allow_html=True)
st.markdown('<div class="footer"><p></p></div>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    #GithubIcon {
      visibility: hidden;
    }
    </style>
    """,unsafe_allow_html=True)
st.markdown('<div id="GithubIcon"><p></p></div>', unsafe_allow_html=True)
