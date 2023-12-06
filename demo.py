import streamlit as st
import openai
import os
import time
from dotenv import load_dotenv

import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,HypotheticalDocumentEmbedder
from tqdm import tqdm
import tiktoken
from io import StringIO
import re
import pickle
from PyPDF2 import PdfReader
import logging
logging.basicConfig(level=logging.INFO) 
import io

load_dotenv()
# Set your OpenAI API key here
openai.api_key = st.secrets['OPENAI_API_KEY']

OPENAI_API_KEY =  st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]


model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)


analysing_model = OpenAI(model_name="gpt-4")

extracting_model = OpenAI(model_name="gpt-4-1106-preview")

qa_chain = load_qa_chain(analysing_model, chain_type="stuff")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-west4-gcp-free"
)

index_name = "lta-test"



task_info = """You are a customer service for an app that give funding predictions for various agencies,insights,
analysis and trend. The documents include information on government agency budgets,
grant allocations, and financial reports. Provide a breakdown and as detail answer as possible with relevant 
budget numbers for all agencies. You can make predictions where the budgets will go based on the information provided.
The user may ask related or non-related questions, answer to the following user question accordingly:  """



text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# Tell tiktoken what model we'd like to use for embeddings
tiktoken.encoding_for_model('text-embedding-ada-002')

# Intialize a tiktoken tokenizer (i.e. a tool that identifies individual tokens (words))
#tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text: str) -> int:
    """
    Split up a body of text using a custom tokenizer.

    :param text: Text we'd like to tokenize.
    """
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def extract_text_from_pdf(pdf_file_path):
    text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text



def chunk_by_section(document):
    print('chunking...')
    text = extract_text_from_pdf(document)
    # Define a regular expression pattern to match section headers
    section_pattern = re.compile(r'SEC\.')
    # Use the pattern to split the content into sections
    sections = re.split(section_pattern, text)

    # Remove empty strings from the list
    sections = [section.strip() for section in sections if section.strip()][1:]
    print(len(sections))
    selected_sections = [section for section in sections]
    
    return selected_sections


def chunk_by_size(text: str, size: int = 200) -> list[Document]:
    """
    Chunk up text recursively.
    
    :param text: Text to be chunked up
    :return: List of Document items (i.e. chunks).|
    """
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = size,
    chunk_overlap = 20,
    length_function = tiktoken_len,
    add_start_index = True,
)
    return text_splitter.create_documents([text])

def divide_into_groups(chunks):
    groups = []
    start_index = 0
    group_size = 2  # Initial group size

    while start_index < len(chunks):
        group = chunks[start_index:start_index + group_size]
        groups.append(group)
        start_index += group_size
        group_size *= 2  # Double the group size for the next iteration
    
    last_group = len(groups[-1]) 
    before_last = len(groups[-2])
    if last_group < (before_last//2):
        groups[-2] = groups[-1]+groups[-2]
        groups = groups[:-1]
    elif last_group < (before_last):
        temp = groups[-1]
        groups[-1] = groups[-2]
        groups[-2] = temp

    return groups

def merge_with_numbering(texts):
    merged_text = ""

    for i, text in enumerate(texts, start=1):
        merged_text += f"{i}. {text}\n"

    return merged_text

def chunk_summary(docs, namespace, depth):

    print('funding summary info extraction in progress...')
    summary_prompt = PromptTemplate(
        input_variables = ['chunk'],
        template = "Extract the funding related information from the following documents. The information will be used to funding analysis and predictions. Documents: {chunk}"
    )

    chunks = divide_into_groups(docs)
    chain = LLMChain(llm=extracting_model, prompt=summary_prompt)

    funding_summaries = []
    for chunk in tqdm(chunks):
        summary = chain.run(chunk)
        funding_summaries.append(summary)

    summary = merge_with_numbering(funding_summaries)
    
    if depth=='Extensive':
        with open('summary/'+namespace+'.smry', 'w') as file:
            file.write(summary)


    return summary


def get_similiar_docs(vectorstore, query, namespace, k=9, score=False):
  if score:
    similar_docs = vectorstore.similarity_search_with_score(query, k=k, namespace=namespace)
  else:
    similar_docs = vectorstore.similarity_search(query, k=k, namespace=namespace)
  return similar_docs


def get_answer(query, context):
  #similar_docs = get_similiar_docs(context)
  context_doc = Document(page_content=context)
  answer = qa_chain.run(input_documents=[context_doc], question=query)
  return answer


def get_funding_info(uploaded_file, file_path, namespace, query, depth):

    depth_value = {'Light':14, 'Medium':30, 'Extensive':62}
    if depth=='Memory':
        with open(file_path, 'r') as file:
            funding_summary = file.read()
        logging.info('summary already in the disk, loaded summary:'+file_path+', size:'+str(len(funding_summary)))
    else:
        prompt_template = """Give me funding predictions, Which agencies will be funded.
        Question: {question}
        Answer:"""

        prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
        llm_chain = LLMChain(llm=extracting_model, prompt=prompt)
        hyde_embed = HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=embed)
        index = pinecone.Index(index_name)

        if namespace not in index.describe_index_stats()['namespaces']:
            pdf_file = io.BytesIO(uploaded_file.getvalue())
            # pdf_reader = PdfReader(pdf_file)
            #stringio = StringIO(uploaded_file.getvalue().decode("latin-1"))
            #leg_text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
            #sections = chunk_by_section('../data/'+file_name)
            sections = chunk_by_section(pdf_file)

            #logging.info('length of sections selcted:'+str(len(sections))+' '+sections[0])


            vectorstore = Pinecone.from_texts(sections, hyde_embed, index_name=index_name, namespace=namespace)

        else:
            vectorstore = Pinecone(index, hyde_embed.embed_query, text_field)

        #similar_query = 'Give me a funding prediction and analysis for various health related agencies' 
        print('Doing analysis with:'+depth)
        similar_docs = get_similiar_docs(vectorstore, query, namespace, k=depth_value[depth])
        funding_summary = chunk_summary(similar_docs, namespace, depth)

    return funding_summary

def main():
    '''    st.set_page_config(
         page_title="Visit Recognition",
         page_icon=" ",
         layout="wide",
         initial_sidebar_state="expanded",
    )
    #st.set_page_config(page_title="Visit Recognition", layout="wide")'''
    padding_top =0
    st.markdown(f"""
    <style>
    .reportview-container .main .block-container{{
        padding-top: {padding_top}rem;
    }}
    h1{{
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("# LegiGPT⚖️ ")
    
    options = ( 'Light', 'Medium', 'Extensive')
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False)
        selectbox_placeholder = st.empty()
        depth = selectbox_placeholder.selectbox('Choose depth of analysis', options)


    #st.title("LegiGPT")

    if "messages" not in st.session_state or len(st.session_state["messages"])==0:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How may I assist you today with funding predictions, analysis, and insights?"}] 

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    funding_summary = None
    
    folder_name = 'summary'

    analysis_prompt_template = """You are giving funding predictions for various agencies, insights and analysis on the funding-related 
    entities and trends based on the documents provided.
    The documents include information on government agency budgets, 
    grant allocations, and financial reports. Answer the following question:{question}
    The following were your previous responses, continue from where you left off...:
    "{previous}"
    """
    analysis_prompt = PromptTemplate(input_variables=["question","previous"], template=analysis_prompt_template)
    detail = 3
    
    question_prompt_template = """Q: {question}
    classify the question either specific or broad. only respond either 'broad' or 'specific' only"""

    question_prompt = PromptTemplate(input_variables=["question"], template=question_prompt_template)



    if uploaded_file is not None:

        st.session_state.messages = []
        file_name = uploaded_file.name
        namespace = file_name.split('.')[0]
        file_path = os.path.join(folder_name, namespace+'.smry')
        logging.info(namespace+'-'+file_name)

        if os.path.isfile(file_path):
            options = ('Memory', 'Light', 'Medium', 'Extensive')    
            depth = selectbox_placeholder.selectbox('Choose depth of analysis', options)
        #uploaded_file = None
        #funding_summary = funding_summary_sample

    else:
        logging.info('upload file is none')

    # React to user input
    if user_query := st.chat_input("What is up?", disabled = not uploaded_file) :
        # Display user message in chat message container
        logging.info(user_query)
        st.chat_message("user").markdown(user_query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})

        funding_summary = get_funding_info(uploaded_file, file_path, namespace, user_query, depth) 

        problem_nature = 'Due to the extensive nature of the information, the response you will give here will be just the first part, \
         you\'ll continue from where you left off in the next response '
        

        question_type = analysing_model(question_prompt.format(question=user_query)).lower() 

        logging.info("The Question type is:"+question_type)

        sample_response = 'SAMPLE RESPONSE'

        if question_type == 'specific':
            input_prompt = task_info + user_query
            final_response = get_answer(input_prompt, funding_summary)
            #final_response = sample_response
            with st.chat_message("assistant"):
                st.markdown(final_response)

        else:
            responses = []
            for i in range(detail):
                if not responses:
                    #print('initial')
                    initial_question = task_info + problem_nature + user_query
                    response = get_answer(initial_question, funding_summary)
                    #response = sample_response
                    with st.chat_message("assistant"):
                        st.markdown(response)
                else:
                    previous_response='\n'.join(responses)
                    input_prompt = analysis_prompt.format(question=user_query, previous=previous_response)
                    #print(input_prompt)
                    st.write('loading...')
                    #time.sleep(5)
                    response = get_answer(input_prompt, funding_summary)
                    #response = sample_response+'-'+str(i)
                    st.markdown(response)

                responses.append(response)

            final_response = '\n'.join(responses)

        st.session_state.messages.append({"role": "assistant", "content": final_response})

    else:
        logging.info('funding summary is None')

    #    if user_query:
            #st.text("ChatGPT:")
          
            
            #print(response)
    #       st.write(sample_response)

if __name__ == "__main__":
    main()

