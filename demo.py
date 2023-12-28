import streamlit as st
import logging
# from streamlit_extras.add_vertical_space import add_vertical_space
import openai
import os
import time
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from analysis import Analysis
# load_dotenv()



task_info = """You are a customer service for an app that give funding predictions for various agencies,insights,
analysis and trend. The documents include information on government agency budgets,
grant allocations, and financial reports. Provide a breakdown and as detail answer as possible with relevant 
budget numbers for all agencies. You can make predictions where the budgets will go based on the information provided.
The user may ask related or non-related questions, answer to the following user question accordingly:  """



def main():
    '''    st.set_page_config(
         page_title="Visit Recognition",
         page_icon=" ",
         layout="wide",
         initial_sidebar_state="expanded",
    )
    #st.set_page_config(page_title="Visit Recognition", layout="wide")'''
    st.set_page_config(layout="wide")
    padding_top =0
    # st.markdown(f"""
    # <style>
    # .reportview-container .main .block-container{{
    #     padding-top: {padding_top}rem;
    # }}
    # h1{{
    #     text-align: center;
    # }}
    # </style>
    # """, unsafe_allow_html=True)

    st.markdown('''
    <style>
    @import url('https://fonts.googleapis.com/css?family=Heebo'); 
    @import url('https://fonts.googleapis.com/css?family=Heebo:400,600,800,900');  

    body * { 
        -webkit-font-smoothing: subpixel-antialiased !important; 
        text-rendering:optimizeLegibility !important;
    }

    body hr {
        border-bottom: 1.5px solid rgba(23, 48, 28, 0.5); 
    }

    div[data-testid="stToolbarActions"] {
        visibility:hidden;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    div[data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }

    div.stButton > button:first-child {
        width: 200px;
        background-color: rgba(23, 48, 28, 0.95) ;
        color: #F6F4F0; 
    }
    div.stButton p {
        font-family: "Heebo";
        font-weight:600;
        font-size: 15px;
        letter-spacing: 0.25px;
        padding-top: 1px;
    }

    div.stLinkButton > a:first-child {
        width: 125px;
        background-color: rgba(23, 48, 28, 0.95) ;
        font-family: "Heebo" !important;
        letter-spacing: 0.25px;
        
    }
    div.stLinkButton p {
        font-size: 15px !important;
        color: #F6F4F0;
        font-family: "Heebo" !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] {
        top: 5rem;
        width: 200px !important; 
        background-color:#CDD4D0;
        background: #F6F4F0;
        border-right: 1.5px solid rgba(23, 48, 28, 0.5);
    }
    div[data-testid="collapsedControl"] {
        top:5.15rem;
    }
    div[data-testid="stExpander"] {
        background-color: rgba(247, 250, 248, 0.45) ;
        background: transparent;
        border: 0px solid black;
    }
    .st-emotion-cache-yf5hy5 p:nth-child(1) {
        font-size: 16px;
        color: green;
        font-family: "Georgia";
    }
    .st-emotion-cache-yf5hy5 p:nth-child(2) {
        font-size: 2.25rem;
        font-weight: 800;
        font-family: 'Heebo';
        line-height:1.15;
        letter-spacing: 0.25px;
        margin: 10px 0 0 0;
    }
    header[data-testid="stHeader"] {
        background: url('https://res.cloudinary.com/djjxmauw3/image/upload/v1703595540/aslc1b6vxklzjptc9dfd.png');
        background-size: contain ;
        background-repeat: no-repeat;
        background-color:rgb(23, 48, 28);
        height: 5rem;
    }

    div[data-testid="stAppViewContainer"] > section:nth-child(2) {
        overflow-x: hidden;
    }
    .st-emotion-cache-uf99v8 {
        overflow-x: hidden;
    }

    .appview-container > section:nth-child(2) > div:nth-child(1) {
        padding: 4.5rem 0.5rem 0rem 1rem;
    }
    .appview-container > section:nth-child(1) > div:nth-child(1) > div:nth-child(2) {
        padding: 1rem 1.5rem 1.5rem 1.5rem;
    }
    .st-dn {
        background-color: transparent;
    }


    div[data-testid="textInputRootElement"] {
        border: 1px solid rgba(23, 48, 28, 0.95);
    }
    div[data-testid="stForm"] {
        border: 0px;
        padding:0;
    }
    div[data-testid="stExpanderDetails"] p {
        font-family:'Georgia';
        font-size: 18px;
    }
    div[data-testid="StyledLinkIconContainer"] {
        font-weight: 900;
        font-family:'Heebo';
        font-size: 2.5rem;
        letter-spacing: 0.25px;
    }

    div[data-testid="stExpander"] > details {
        bordder-radius: 0;
        border-color: rgba(255, 255, 255, 0.05);
    }
    div[data-baseweb="tab-panel"] > div:nth-child(1) > div:nth-child(1) {
        gap: 0.5rem;
    }

    div[data-testid="stExpander"] > details > summary:hover {
        color: rgb(23, 48, 28);
    }
    
    div[data-baseweb="select"] {
        font-family: "Heebo";
        font-weight:600;
        font-size: 15px;
        letter-spacing: 0.25px;
    }

    ul[data-testid="stVirtualDropdown"] li {
        text-align: center;
        font-family: "Heebo";
    }
    ul[data-testid="stVirtualDropdown"] li:hover {
        color: rgba(23, 48, 28, 0.95);
        background-color:#B3BCB4;
    }

    div[data-baseweb="select"] > div:first-child > div > div:first-child {
        padding-left: 48px;
        color: #F6F4F0;
        padding-top: 1px;
        
    }

    div[data-baseweb="select"] div {
        background-color: rgba(23, 48, 28, 0.95);
        color: #F6F4F0;
        border: 0px;
    }
    div[data-baseweb="popover"] .st-dk {
        background-color: rgba(23, 48, 28, 0.95);
    }
    div[data-baseweb="popover"] li {
        color: #F6F4F0;
        background-color: rgba(23, 48, 28, 0.95);
    }
    div[data-baseweb="popover"]  .st-emotion-cache-35i14j {
        background: #B3BCB4;
        color: rgba(23, 48, 28, 0.95) !important;
    }


    div[data-baseweb="select"] svg {
        color: #F6F4F0;
    }

    div[data-testid="stForm"] .st-dk {
        background-color: #DFE3E0;
    }

    div[data-testid="stCaptionContainer"] {
        margin-bottom: -1.75rem;
    }

    </style>
    ''', unsafe_allow_html=True)





    # st.markdown("# LegiGPT⚖️ ")
    # gen_analysis = False
    
    options = ( 'Extensive', 'Medium', 'Light' )
    gen_analysis = False
    
    with st.sidebar:
        st.sidebar.header("Document")
        document_name = st.text_input(label="",placeholder='name of the document', label_visibility="collapsed", key="text")
        uploaded_file = st.file_uploader("Upload the document", label_visibility="collapsed", type="pdf", accept_multiple_files=False)

        st.sidebar.header("Analysis")
        analysis_selectbox_placeholder = st.empty()
        analysis_option = analysis_selectbox_placeholder.selectbox(label='select', label_visibility= "collapsed", options=('Generate', 'Chat'))
        depth_selectbox_placeholder = st.empty()
        depth = depth_selectbox_placeholder.selectbox(label='select', label_visibility= "collapsed", options=options)
       

        
        if analysis_option == 'Generate' and st.sidebar.button(' 📝 Generate analysis', use_container_width = True, disabled = not uploaded_file or not document_name):
            gen_analysis = True

    
        


    #st.title("LegiGPT")

    if "messages" not in st.session_state or len(st.session_state["messages"])==0:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How may I assist you today with funding predictions, analysis, and insights?"}] 

    # Display chat messages from history on app rerun
    # if analysis=='Chat':
    #     for message in st.session_state.messages:
    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])

    funding_summary = None
    
    folder_name = 'summary'

    analysis_prompt_template = """You are giving funding predictions for various agencies and programs and insights and analysis on the funding-related 
    entities and trends based on the documents provided.
    The documents include information on government agency budgets, 
    grant allocations, and financial reports. Answer the following question:{question}
    The following were your previous responses, continue from where you left off...:
    "{previous}"
    """
    analysis_prompt = PromptTemplate(input_variables=["question","previous"], template=analysis_prompt_template)
    detail = 3



    if uploaded_file is not None:

        st.session_state.messages = []
        file_name = uploaded_file.name
        namespace = file_name.split('.')[0]
        if analysis_option=='Chat':
            file_path = os.path.join(folder_name, namespace+'.smry')
            logging.info(namespace+'-'+file_name)

            if os.path.isfile(file_path):
                options = ('Memory', 'Light', 'Medium', 'Extensive')    
                depth = depth_selectbox_placeholder.selectbox(label='select', label_visibility= "collapsed", options=options)
        else:
            sections = ['intro', 'provisions', 'states']
            section_paths = [os.path.join(folder_name, namespace+'-'+section+'.smry') for section in sections]
            summary_exist = [os.path.isfile(file_path) for file_path in section_paths]
            if any(summary_exist):
                print('summary exists')
                options = ('Memory', 'Light', 'Medium', 'Extensive')    
                depth = depth_selectbox_placeholder.selectbox('Choose depth of analysis', options)



        #uploaded_file = None
        #funding_summary = funding_summary_sample

    else:
        logging.info('upload file is none')

    
    analysis = Analysis()


    if analysis_option == 'Generate':
        st.markdown(f"<h1 style='text-align: center; color: black; font-size: 64px;'>{document_name}</h1>", unsafe_allow_html=True)
        if not gen_analysis:
            st.markdown('Generating a comphrensive analysis can take a while :)')
        else:
            comprhensive_analysis = analysis.generate_analysis(document_name, uploaded_file, depth, sample={'intro':False, 'states': False, 'provisions': False, 'fiscal': False, 'random':False})

            intro_response = comprhensive_analysis['intro']
            states_response = comprhensive_analysis['states']
            provisions_response = comprhensive_analysis['provisions']
            fiscal_response = comprhensive_analysis['fiscal']
            random_response = comprhensive_analysis['random']

            col1, col2 = st.columns([0.5,0.5])
            with col1:
                row1_spacer1, row1_1, row1_spacer2 = st.columns((0.001, 0.9, 0.001))
                with row1_1:
                    # add_vertical_space()
                    st.markdown(intro_response)

                section1_spacer1, section1_1, section1_spacer2 = st.columns((0.001, 0.9, 0.001))

                with section1_1:
                    st.header('Spending Provisions')
                    for response in provisions_response:
                            st.subheader(response['agency'])
                            st.markdown(response['analysis'])
                            


                section2_spacer1, section2, section2_spacer2 = st.columns((0.001, 0.9, 0.001))

                with section2:
                    st.header('Funding distribution across states')
                    st.markdown(states_response)

                section3_spacer1, section3, section3_spacer3 = st.columns((0.001, 0.9, 0.001))

                with section3:
                    st.header('Funding Allocation Across Diverse Fiscal Years')
                    st.markdown(fiscal_response)

                
            
            with col2:
                section4_spacer1, section4_1, section4_spacer2 = st.columns((0.001, 0.9, 0.001))
                with section4_1:
                    for topics in random_response:
                        for topic, topic_analysis in topics.items():
                                st.header(topic)
                                st.markdown(topic_analysis)



    else:
    # React to user input
        if user_query := st.chat_input("What is up?", disabled = not uploaded_file) :
            # Display user message in chat message container
            logging.info(user_query)
            st.chat_message("user").markdown(user_query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})

            funding_summary = analysis.get_funding_info(uploaded_file, file_path, namespace, user_query, depth) 

            problem_nature = 'Due to the extensive nature of the information, the response you will give here will be just the first part, \
            you\'ll continue from where you left off in the next response '
            

            question_type = analysis.classify_question(user_query)

            logging.info("The Question type is:"+question_type)


            if question_type == 'specific':
                input_prompt = task_info + user_query
                final_response = analysis.get_answer(input_prompt, funding_summary)
                #final_response = sample_response
                with st.chat_message("assistant"):
                    st.markdown(final_response)

            else:
                responses = []
                for i in range(detail):
                    if not responses:
                        #print('initial')
                        initial_question = task_info + problem_nature + user_query
                        response = analysis.get_answer(initial_question, funding_summary)
                        with st.chat_message("assistant"):
                            st.markdown(response)
                    else:
                        previous_response='\n'.join(responses)
                        input_prompt = analysis_prompt.format(question=user_query, previous=previous_response)
                        st.write('loading...')
                        response = analysis.get_answer(input_prompt, funding_summary)
                        st.markdown(response)

                    responses.append(response)

                final_response = '\n'.join(responses)

            st.session_state.messages.append({"role": "assistant", "content": final_response})

        else:
            logging.info('funding summary is None')


if __name__ == "__main__":
    main()

