import streamlit as st
import openai
import os
from dotenv import load_dotenv

import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.base import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tqdm import tqdm
import tiktoken
from io import StringIO
import re
import pickle
from PyPDF2 import PdfReader

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


model_name = "gpt-4"
llm = OpenAI(model_name=model_name)

qa_chain = load_qa_chain(llm, chain_type="stuff")

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-west4-gcp-free"
)

index_name = "lta-test"



task_info = """You are a customer service for an app that give funding predictions for various agencies,insights,
analysis and trends for a comapny in interested in health related RFPs based on the documents provided.
The documents include information on government agency budgets,
grant allocations, and financial reports. The user may ask related or non-related questions, answer to the following user question accordingly:  """

sample_response = '''The documents highlight multiple entities receiving allocations, with an explicit emphasis on health-related programs and initiatives. 

1. Department of Health and Human Services (HHS): The HHS received significant amounts for a variety of health-focused activities. In 2010, it was authorized to receive $100,000,000 for infrastructure expansion on healthcare facilities for research, inpatient tertiary care, and outpatient clinical services at academic health centers. Moreover, significant funds have been marked for community health centers and the National Health Service Corps Fund (CHC Fund).

2. National Health Service Corps: This entity, within the HHS, received a significant escalating allocation each fiscal year starting from $320,461,632 in 2010, gradually increasing to $1,154,510,336 by 2015. 

3. Prevention and Public Health Fund: Allocated an escalating amount starting from $500,000,000 in 2010, which increases to $2,000,000,000 in 2015 and each fiscal year thereafter.

4. Community Health Center Fund (CHC Fund): This fund mainly focuses on expanded and sustained national investment in community health centers and the National Health Service Corps, with funds increasing from $1,000,000,000 in 2011 to $3,600,000,000 in 2015.

Considering these allocations, the Department of Health and Human Services, the National Health Service Corps, and various health and prevention funds are likely recipients of funding. 

In terms of Requests for Proposals (RFPs), one could anticipate focus areas including debt service on, or the direct construction or renovation of healthcare facilities, public health and prevention programs, community health centers, and health services corps. There might be special emphasis on providing greater access to health care within States, and supporting the financial viability of the State’s public medical and dental school as well as its academic health centers.

Favorable project categories for funding would likely include health infrastructure construction and renovation, health services program expansion, community preventive health activities aimed at reducing chronic disease rates and health disparities, and professional development in the health sector. Projects that address social determinants of health, disparities in health access and outcomes, and those focusing on prevention would also be prioritized.
'''
with open('funding_summary.txt', 'r') as file:
    # Read the content of the file
    funding_summary_sample = file.read()



text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# Tell tiktoken what model we'd like to use for embeddings
tiktoken.encoding_for_model('text-embedding-ada-002')

# Intialize a tiktoken tokenizer (i.e. a tool that identifies individual tokens (words))
tokenizer = tiktoken.get_encoding('cl100k_base')

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
    token_limit = 6000
    text = extract_text_from_pdf(document)
    # Define a regular expression pattern to match section headers
    section_pattern = re.compile(r'SEC\.')
    # Use the pattern to split the content into sections
    sections = re.split(section_pattern, text)

    # Remove empty strings from the list
    sections = [section.strip() for section in sections if section.strip()][1:]
    print(len(sections))
    selected_sections = [section for section in sections if tiktoken_len(section) < token_limit]
    
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

def divide_into_groups(documents, k):
    # Calculate the total number of groups needed
    num_groups = len(documents) // k + (len(documents) % k > 0)

    # Create empty groups
    groups = [[] for _ in range(num_groups)]

    # Distribute documents into groups
    for i, doc in enumerate(documents):
        group_index = i % num_groups
        groups[group_index].append(doc)

    return groups

def merge_with_numbering(texts):
    merged_text = ""

    for i, text in enumerate(texts, start=1):
        merged_text += f"{i}. {text}\n"

    return merged_text

def chunk_summary(docs, namespace):

    print('funding summary info extraction in progress...')
    summary_prompt = PromptTemplate(
        input_variables = ['chunk'],
        template = "Summarize the funding related information from the following document. The summary will be used to funding analysis and predictions. Document: {chunk}"
    )

    chunks = divide_into_groups(docs, k=3)
    chain = LLMChain(llm=llm, prompt=summary_prompt)

    funding_summaries = []
    for chunk in tqdm(chunks):
        summary = chain.run(chunk)
        funding_summaries.append(summary)

    summary = merge_with_numbering(funding_summaries)

    with open('summary/'+namespace+'.smry', 'w') as file:
        file.write(summary)


    return summary


def get_similiar_docs(query, namespace, k=9, score=False):
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
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")


    #st.title("LegiGPT")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    funding_summary = None
    
    folder_name = 'summary'

    if uploaded_file is not None:

        st.session_state.messages = []
        file_name = uploaded_file.name
        namespace = file_name.split('.')[0]
        file_path = os.path.join(folder_name, namespace+'.smry')
        print(namespace, file_name)

        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                funding_summary = file.read()
            print('summary already in the disk')
        else:
            #stringio = StringIO(uploaded_file.getvalue().decode("latin-1"))
            #leg_text = stringio.read()
            sections = chunk_by_section('../data/'+file_name)

            print('length of sections selcted', len(sections))

            with open('hyde_embedding.pkl', 'rb') as f:
                hyde_embed = pickle.load(f)

            vectorstore = Pinecone.from_texts(sections, hyde_embed, index_name=index_name, namespace=namespace)

            similar_query = 'Give me a funding prediction and analysis for various health related agencies' 
            similar_docs = get_similiar_docs(similar_query, namespace, k=45)
            funding_summary = chunk_summary(similar_docs, namespace)
        #funding_summary = funding_summary_sample

    # React to user input
    if funding_summary is not None: 
        if user_query := st.chat_input("What is up?") :
            # Display user message in chat message container
            print(user_query)
            st.chat_message("user").markdown(user_query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})

            
            input_prompt = task_info + user_query
            response = get_answer(input_prompt, funding_summary)


            #response = sample_response
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    #    if user_query:
            #st.text("ChatGPT:")
          
            
            #print(response)
    #       st.write(sample_response)

if __name__ == "__main__":
    main()

