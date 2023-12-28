import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,HypotheticalDocumentEmbedder
from tqdm import tqdm
from io import StringIO
import pickle
from PyPDF2 import PdfReader
import logging
logging.basicConfig(level=logging.INFO) 
import io
import openai
import streamlit as st
from sample_response import sample_provisions, sample_introduciton, sample_states, sample_random
import ast
import re
import os
from dotenv import load_dotenv

class Analysis:

    def __init__(self) -> None:
        load_dotenv()
        openai.api_key = st.secrets['OPENAI_API_KEY']

        openai_api_key =  st.secrets['OPENAI_API_KEY']
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]


        model_name = 'text-embedding-ada-002'

        self.embed = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=openai_api_key
        )


        self.analysing_model = OpenAI(model_name="gpt-4")

        self.extracting_model = OpenAI(model_name="gpt-4-1106-preview")

        self.qa_chain = load_qa_chain(self.analysing_model, chain_type="stuff")

        pinecone.init(
            api_key=pinecone_api_key,
            environment="us-west4-gcp-free"
        )

        self.index_name = "lta-test"
        self.summary_folder_name = 'summary'
        
    def extract_text_from_pdf(self, pdf_file_path):
        text = ""
        pdf_reader = PdfReader(pdf_file_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def chunk_by_section(self, document):
        print('chunking...')
        text = self.extract_text_from_pdf(document)
        # Define a regular expression pattern to match section headers
        section_pattern = re.compile(r'SEC\.')
        # Use the pattern to split the content into sections
        sections = re.split(section_pattern, text)

        # Remove empty strings from the list
        sections = [section.strip() for section in sections if section.strip()][1:]
        print(len(sections))
        selected_sections = [section for section in sections]
        
        return selected_sections

    def divide_into_groups(self, chunks):
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

    def merge_with_numbering(self, texts):
        merged_text = ""

        for i, text in enumerate(texts, start=1):
            merged_text += f"{i}. {text}\n"

        return merged_text

    def chunk_summary(self, docs, section):

        print('summarizing the chunks in progress...')

        section_prompt = {'default':'',
                          'provisions':'for different agencies',
                          'states': 'across diffrent states or regions ',
                          'fiscal': 'across different fiscal years'}
        
        if section not in section_prompt:
            section = 'default'
        
 
        summary_prompt = PromptTemplate(
            input_variables = ['chunk'],
            template = "Extract the funding related information "+section_prompt[section] +" from the following documents. The information will be used to funding analysis and predictions. Documents: {chunk}"
        )

        chunks = self.divide_into_groups(docs)
        chain = LLMChain(llm=self.extracting_model, prompt=summary_prompt)

        funding_summaries = []
        for chunk in tqdm(chunks):
            summary = chain.run(chunk)
            funding_summaries.append(summary)

        summary = self.merge_with_numbering(funding_summaries)
    
        return summary


    def get_similiar_docs(self, vectorstore, query, namespace, k=9, score=False):
        logging.info('Extracting related documents from the database...')
        if score:
            similar_docs = vectorstore.similarity_search_with_score(query, k=k, namespace=namespace)
        else:
            similar_docs = vectorstore.similarity_search(query, k=k, namespace=namespace)
        return similar_docs


    def get_answer(self, query, context):
        #similar_docs = get_similiar_docs(context)
        context_doc = Document(page_content=context)
        answer = self.qa_chain.run(input_documents=[context_doc], question=query)
        return answer


    def get_funding_info(self, uploaded_file, section, namespace, query, depth):

        depth_def = {'Light':6, 'Medium':30, 'Extensive':62}
        
        
        if depth=='Memory':
            file_path = os.path.join(self.summary_folder_name, namespace+'-'+section+'.smry') 
            if not os.path.isfile(file_path):
                depth='Extensive'
            else:
                with open(file_path, 'r') as file:
                    funding_summary = file.read()
                logging.info('summary already in the disk, loaded summary:'+file_path+', size:'+str(len(funding_summary)))
                return funding_summary

        prompt_template = """Give me funding predictions, Which agencies will be funded.
        Question: {question}
        Answer:"""

        prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
        llm_chain = LLMChain(llm=self.extracting_model, prompt=prompt)
        hyde_embed = HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=self.embed)
        index = pinecone.Index(self.index_name)

        if namespace not in index.describe_index_stats()['namespaces']:
            pdf_file = io.BytesIO(uploaded_file.getvalue())
            # pdf_reader = PdfReader(pdf_file)
            #stringio = StringIO(uploaded_file.getvalue().decode("latin-1"))
            #leg_text = uploaded_file.getvalue().decode("utf-8", errors='ignore')
            #sections = chunk_by_section('../data/'+file_name)
            sections = self.chunk_by_section(pdf_file)

            #logging.info('length of sections selcted:'+str(len(sections))+' '+sections[0])


            vectorstore = Pinecone.from_texts(sections, hyde_embed, index_name=self.index_name, namespace=namespace)

        else:
            vectorstore = Pinecone(index, hyde_embed.embed_query, "text")

        #similar_query = 'Give me a funding prediction and analysis for various health related agencies' 
        print('Doing analysis with:'+depth, query)
        similar_docs = self.get_similiar_docs(vectorstore, query, namespace, k=depth_def[depth])
        funding_summary = self.chunk_summary(similar_docs, section)
        if depth=='Extensive':
            with open(file_path, 'w') as file:
                file.write(funding_summary)
            print('saved the funding summary as '+file_path)

            
        

        return funding_summary


    def classify_question(self, question):
        question_prompt_template = """Q: {question}
        Classify the question either specific or broad. only respond either 'broad' or 'specific' only"""

        question_prompt = PromptTemplate(input_variables=["question"], template=question_prompt_template)
        question_type = self.analysing_model(question_prompt.format(question=question)).lower() 

        logging.info("The Question type is:"+question_type)

        return question_type
    
    def generate_random_topics(self, namespace, document_name, uploaded_file, depth):
        prompt = f"""You are generating topics to be disscussed in a blog on funding analysis of agencies for {document_name} the topics Spending Provisions, Funding distribution across states and Funding Allocation Across Diverse Fiscal Year
          have already discussed in the blog. what other topics other than those topics would you add to the blog. Give the name of the topics as a python list of dictionaries in the following format. 
        [{{"topic": "Spending Provisions", "description":"Which agencies receive funding? which programs or projects under the agencies receive funding?"}},
        {{"topic":"Funding distribution across states", "description":"Disuss funding distribution across states or regions"}},
        {{"topic":"Funding distribution across states", "description":"Disuss funding distribution across states or regions"}},
        {{"topic":"Funding Allocation Across Diverse Fiscal Years", "description": "Analyse the funding allocation across diverse fiscal years."}}]
          
          
          start your respone with response = [{{"topic":"..."""
        similar_docs_prompt = "Which agencies receive funding? which programs or projects under the agencies receive funding?"

        file_path = os.path.join(self.summary_folder_name, namespace+'-'+'generated'+'.smry') 
        if depth=='Memory' and os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                response = file.read() 
        else:
            logging.info('The promt for variable topics: '+prompt)
            states_funding_summary = self.get_funding_info(uploaded_file, 'default', namespace, similar_docs_prompt, depth) 
            response = self.get_answer(prompt, states_funding_summary)
            logging.info('Variable topics resposne: '+response)
            if depth=='Extensive' or depth=='Memory':
                with open(file_path, 'w') as file:
                    file.write(response)


        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            topics_dict = ast.literal_eval(match.group(0))
        else:
            logging.warning('No match response found in dictionary format')

        topics_analysis = []
        for topics in topics_dict:
            topic = topics['topic']
            description = topics['description']
            funding_summary = self.get_funding_info(uploaded_file, topic, namespace, description, depth) # depth should Extensive for a good analysis
            analysis_prompt = f"""You're writing part of a blog that {description}. Discuss it based the documents provided below."""
            logging.info("Analysis prompt:"+analysis_prompt)
            analysis_response = self.get_answer(analysis_prompt, funding_summary)
            topics_analysis.append({topic : analysis_response})
        
        # logging.info("Random topics analysis:"+topics_analysis)


        return topics_analysis


    def generate_analysis(self, document_name, uploaded_file, depth, sample):
        sections_prompt = {'Introduction': f"You are writing introduction part of a blog on funding analysis of agencies for {document_name}. The blog discusses funding of agencies, how funding distribute in different states... write the introduction part of the blog. start with: The {document_name}...",
                            'Spending Provisions':'Which agencies receive funding? which programs or projects under the agencies receive funding?',
                            'Funding distribution across states': 'Disuss funding distribution across states or regions Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New Hampshire New Jersey New Mexico New York North Carolina North Dakota Ohio Oklahoma Oregon Pennsylvania Rhode Island South Carolina South Dakota Tennessee Texas Utah Vermont Virginia Washington West Virginia Wisconsin Wyoming',
                            'Funding Allocation Across Diverse Fiscal Years': 'Analyse the funding allocation across diverse fiscal years.'
                            }
        attach_document_prompt = f" \n The following documents provided are from {document_name}: "
        # if sample:
        #     comprhensive_analysis = {'intro': sample_introduciton,
        #                              'states': sample_states,
        #                              'provisions':sample_provisions}
        #     return comprhensive_analysis
        
        file_name = uploaded_file.name
        namespace = file_name.split('.')[0]

        if sample['intro']:
            intro_response = sample_introduciton

        else:
            intro_funding_summary = self.get_funding_info(uploaded_file, 'intro', namespace, sections_prompt['Introduction'], depth) 
            intro_response = self.get_answer(sections_prompt['Introduction']+attach_document_prompt, intro_funding_summary)             
        
        if sample['states']:
            states_response = sample_states
        else:
            states_funding_summary = self.get_funding_info(uploaded_file, 'states', namespace, sections_prompt['Funding distribution across states'], depth) 
            states_prompt = f"You're writing part of a blog that discusses the funding distribution of states {document_name} based on the documents provided below, Dicuss the funding distribution across states or regions."
            states_response = self.get_answer(states_prompt+attach_document_prompt, states_funding_summary)
        
        if sample['states']:
            states_response = sample_states
        else:
            states_funding_summary = self.get_funding_info(uploaded_file, 'states', namespace, sections_prompt['Funding distribution across states'], depth) 
            states_prompt = f"You're writing part of a blog that discusses the funding distribution of states {document_name} based on the documents provided below, Dicuss the funding distribution across states or regions."
            states_response = self.get_answer(states_prompt+attach_document_prompt, states_funding_summary)

        if sample['fiscal']:
            fiscal_response = sample_states
        else:
            fiscal_funding_summary = self.get_funding_info(uploaded_file, 'fiscal', namespace, sections_prompt['Funding Allocation Across Diverse Fiscal Years'], depth) 
            fiscal_prompt = f"You're writing part of a blog that discusses the funding allocation across different fiscal years {document_name} based on the documents provided below, Dicuss the funding in different fiscal years."
            fiscal_response = self.get_answer(fiscal_prompt+attach_document_prompt, fiscal_funding_summary)
            

        if sample['provisions']:
            provisions_resposne = sample_provisions
        else:
            provisions_funding_summary = self.get_funding_info(uploaded_file, 'provisions', namespace, sections_prompt['Spending Provisions'], depth) 
            detail = 0
            provisions_resposne = []
            while detail<7:
                if not provisions_resposne:
                    format_prompt = f"""
                    You are extracting agencies and analysing funding of agencies under {document_name},put the analysis in detail as much as possible with relevant budget numbers for the agencies including their programs and projects when available. Discuss each programs and projects under the agency separately if possible. An analysis of around of 750 words is expected for each agency. You can add your insights and predictions to the analysis.
                    . Due to the extensive nature of the information, the response you will give here will be just the first part which could include only some of the agencies.
                    Return the answer as a python list of dictionaries in the following format, :
                    response = [{{"agency": "Abcd",
                    "analysis": "Includes a series of programs funded by the ACA to boost the effectiveness and efficiency of the Abcd program. 
                    This includes the creation of the Center for Abcd and Abde Innovation (CMI), which is intended to conduct research and demonstration projects to improve
                    efficiency and quality in Abcd, Abde and ADES. CMI was appropriated approximately $10 billion over 10 years (§ 3021). 
                    The ACA also funds other Abcd programs, including the Abcd Independence at Home Demonstration for $30 million over six years (§ 3024). "}}, 
                    {{"agency": "Abde",
                    "analysis": " Includes grants programs focused on the health of enrollees in Abde and the Children’s Health Insurance Program (ADES).
                    Examples of ACA-funded programs include Abde Prevention and Wellness Incentives for $100 million over five years (§ 4108) 
                    and the ADES Childhood Obesity Demonstration for $25 million over five years (§ 4306)."}}] 
                    start with: response = [{{"agency":"...."""
                    final_prompt = sections_prompt['Spending Provisions']+format_prompt+attach_document_prompt
                    logging.info(f'The prompt input is:{final_prompt}')
                    response = self.get_answer(final_prompt, provisions_funding_summary).replace('\'', '').replace('$', '\\$')
                    # Define a regular expression pattern to match newlines within double quotes
                    pattern = re.compile(r'"(.*?)"', re.DOTALL)

                    # Replace newlines within double quotes with spaces
                    response = re.sub(pattern, lambda match: match.group(0).replace('\n', '\\n'), response)

                else:
                    agency_names = [entry['agency'] for entry in provisions_resposne]
                    
                    prev_response = ', '.join(agency_names[:-1]) + ', and ' + agency_names[-1]
                    logging.info(f'agencies {prev_response} are discussed.')
                    continue_prompt = f"""You are extracting agencies and analysing funding of agencies under {document_name},put the analysis in detail as much as possible with relevant budget numbers for the agencies including their programs and projects when available. Discuss each programs and projects under the agency separately if possible. An analysis of around of 750 words is expected for each agency. You can add your insights and predictions to the analysis.
                    . Due to the extensive nature of the information, the response you will give here could only include some of the agencies.

                    Return the answer a python list of dictionaries in the following format, :
                    response = [{{"agency": "Abcd",
                    "analysis": "Includes a series of programs funded by the ACA to boost the effectiveness and efficiency of the Abcd program. 
                    This includes the creation of the Center for Abcd and Abde Innovation (CMI), which is intended to conduct research and demonstration projects to improve
                    efficiency and quality in Abcd, Abde and ADES. CMI was appropriated approximately $10 billion over 10 years (§ 3021). 
                    The ACA also funds other Abcd programs, including the Abcd Independence at Home Demonstration for $30 million over six years (§ 3024). "}}, 
                    {{"agency": "Abde",
                    "analysis": " Includes grants programs focused on the health of enrollees in Abde and the Children’s Health Insurance Program (ADES).
                    Examples of ACA-funded programs include Abde Prevention and Wellness Incentives for $100 million over five years (§ 4108) 
                    and the ADES Childhood Obesity Demonstration for $25 million over five years (§ 4306)."}}]

                    {prev_response} are already discussed and should not be included in this response. start with: response = [{{"agency":"...."""

                    final_prompt = sections_prompt['Spending Provisions']+continue_prompt+attach_document_prompt
                    logging.info(f'The prompt input is:{final_prompt}')
                    response = self.get_answer(final_prompt, provisions_funding_summary).replace('\'', '').replace('$', '\\$')
                    # Define a regular expression pattern to match newlines within double quotes
                    pattern = re.compile(r'"(.*?)"', re.DOTALL)

                    # Replace newlines within double quotes with spaces
                    response = re.sub(pattern, lambda match: match.group(0).replace('\n', '\\n'), response)

                # print(response)

                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    detail+=1
                    response_dict = ast.literal_eval(match.group(0))
                    provisions_resposne.extend(response_dict)
                else:
                    logging.warning('No match response found in dictionary format. detail -' + str(detail))

                print(provisions_resposne)

        if sample['random']:
            random_topics = sample_random
        else:
            random_topics = self.generate_random_topics(namespace, document_name, uploaded_file, depth)

            # logging.info('The topics generated:'+str(random_topics))

        

        comprhensive_analysis = {'intro':intro_response,
                                 'states': states_response,
                                 'provisions':provisions_resposne,
                                 'fiscal':fiscal_response,
                                 'random':random_topics}
        
        return comprhensive_analysis

        