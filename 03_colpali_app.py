import os
import streamlit as st
from streamlit_chat import message
import modal
import base64
from groq import Groq
import time

# Environment setup
os.environ["MODAL_TOKEN_ID"] = st.secrets["MODAL_TOKEN_ID"]
os.environ["MODAL_TOKEN_SECRET"] = st.secrets["MODAL_TOKEN_SECRET"]
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Initialize Modal client
try:
    with modal.Client() as client:
        print("Authentication successful!")
except Exception as e:
    print(f"Authentication failed: {e}")

Companion = modal.Cls.lookup("colpali_rag", "Companion")
companion = Companion()

# Setup upload directory
upload_dir = "./doc"
os.makedirs(upload_dir, exist_ok=True)

# Initialize session state
if 'doc_indexed' not in st.session_state:
    st.session_state['doc_indexed'] = False
if 'current_doc_id' not in st.session_state:
    st.session_state['current_doc_id'] = None
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Greetings! Please feel free to ask any questions related to the uploaded document."]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Streamlit UI
st.title("ColPali RAG")

def process_uploaded_file(uploaded_file):
    """Process the uploaded file and create index"""
    save_path = os.path.join(upload_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with open(save_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Generate a unique doc_id based on filename and timestamp
    doc_id = f"doc_{uploaded_file.name}_{int(time.time())}"
    
    try:
        # Create index for the document
        doc_id = companion.index_document.remote(doc=pdf_bytes, doc_id=doc_id)
        st.session_state['doc_indexed'] = True
        st.session_state['current_doc_id'] = doc_id
        st.success("Document indexed successfully!")
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")
        st.session_state['doc_indexed'] = False
        st.session_state['current_doc_id'] = None

# File uploader
uploaded_file = st.file_uploader("Choose a Document", type=["pdf"])

# Only process the file if it's newly uploaded
if uploaded_file and not st.session_state['doc_indexed']:
    process_uploaded_file(uploaded_file)

# Chat interface
if st.session_state['doc_indexed']:
    # Container for chat history
    response_container = st.container()
    # Container for text box
    text_container = st.container()
    
    with text_container:
        text_query = st.chat_input("Enter your query")
        
        
        if text_query:
                try:
                    # Get search results using existing index
                    results = companion.search.remote(query=text_query, k=1)
                
                    if results and len(results) > 0:
                        # Convert bytes to base64 string
                        content_payload = [{"type": "text", "text": text_query}]
                        content_payload.append({"type": "text", "text": "Answer the question based on the images"})
                        for image_data in results:
                           # Convert bytes to base64 string
                           image_base64 = base64.b64encode(image_data).decode('utf-8')
                           content_payload.append(
                           {
                             "type": "image_url",
                             "image_url": {
                             "url": f"data:image/png;base64,{image_base64}"
                            }
                           }
                         )
                        
                        try:
                            response = client.chat.completions.create(
                                model="llama-3.2-11b-vision-preview",
                                messages=[
                                 {
                                  "role": "user",
                                  "content": content_payload,
                                 }
                                 ],
                                temperature=0.6,
                                max_tokens=1024,
                                top_p=1,
                                stream=False,
                                stop=None,
                            )
                            
                            if response and response.choices:
                                ans = response.choices[0].message.content
                                st.session_state.responses.append(f"User: {text_query}")
                                st.session_state.responses.append(f"Bot: {ans}")
                            
                        except Exception as e:
                            st.error(f"Error making API request: {str(e)}")
                    else:
                        st.warning("No results found for your query.")
                except Exception as e:
                    st.error(f"Error searching document: {str(e)}")

    # Style for chat messages
    st.markdown(
        """
        <style>
        [data-testid="stChatMessageContent"] p{
            font-size: 1rem;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    # Display chat history
    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                if i % 2 == 0:  # Even indices for bot, odd indices for user
                    with st.chat_message('Momos', avatar='icon.png'):
                        st.write(st.session_state['responses'][i])
                else:
                    message(st.session_state['responses'][i], is_user=True, key=str(i) + '_user')

# Add a button to clear the current index and upload a new document
if st.session_state['doc_indexed']:
    if st.button("Clear Current Document"):
        st.session_state['doc_indexed'] = False
        st.session_state['current_doc_id'] = None
        st.session_state['responses'] = ["Greetings! Please feel free to ask any questions related to the uploaded document."]
        st.session_state['requests'] = []
        st.experimental_rerun()