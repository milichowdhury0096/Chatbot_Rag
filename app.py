import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
import os

# Streamlit App Configuration
st.set_page_config(layout="wide")
st.title("Emplochat")

# Sidebar for API Key input
with st.sidebar:
    API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

# Define the embedding function
class OpenAIEmbeddingFunction:
    def __call__(self, texts):
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        embeddings = [embedding['embedding'] for embedding in response['data']]
        return embeddings

# Initialize OpenAI client
class OpenAIClient:
    def __init__(self, api_key):
        openai.api_key = api_key

    def chat(self, *args, **kwargs):
        return openai.ChatCompletion.create(*args, **kwargs)

# Initialize the OpenAI client
client = OpenAIClient(API_KEY)

persist_directory = '/mount/src/Chatbot_multiagent/embeddings'

# Initialize the Chroma DB client
store = Chroma(persist_directory=persist_directory, collection_name="Capgemini_policy_embeddings")

embed_prompt = OpenAIEmbeddingFunction()

# Define the embedding retrieval function
def retrieve_vector_db(query, n_results=2):
    embedding_vector = embed_prompt([query])[0]
    similar_embeddings = store.similarity_search_by_vector_with_relevance_scores(embedding=embedding_vector, k=n_results)
    results = []
    prev_embedding = []
    for embedding in similar_embeddings:
        if embedding not in prev_embedding:
            results.append(embedding)
        prev_embedding = embedding
    return results

# Set the fine-tuned model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "ft:gpt-3.5-turbo-0125:personal::A9eKNr3q"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Enter your query here?"):
    retrieved_results = retrieve_vector_db(query, n_results=3)
    context = ''.join([doc[0].page_content for doc in retrieved_results[:2]])

    # Construct prompt for fine-tuned model
    fine_tuned_prompt = f'''
    You are an expert in Capgemini policies. Provide a detailed answer based on the context provided.
    
    Question: {query}
    
    Context: {context}
    '''

    st.session_state.messages.append({"role": "user", "content": query})

    # Display the user message
    with st.chat_message("user"):
        st.markdown(query)

    # --- 1. Fine-Tuned Model Response ---
    with st.chat_message("assistant"):
        response = client.chat(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": fine_tuned_prompt}],
            stream=False
        )
        fine_tuned_response = response['choices'][0]['message']['content']
        st.markdown(fine_tuned_response)

    # Append the fine-tuned model response to chat history
    st.session_state.messages.append({"role": "assistant", "content": fine_tuned_response})

    # --- 2. Normal RAG Response ---
    # Generate a normal RAG response using the same context
    with st.chat_message("assistant"):
        normal_rag_prompt = f'''
        You are an expert in Capgemini policies. Give a detailed answer based on the context provided.
        
        Question: {query}
        
        Context: {context}
        '''
        response_rag = client.chat(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": normal_rag_prompt}],
            stream=False
        )
        normal_rag_response = response_rag['choices'][0]['message']['content']
        st.markdown(normal_rag_response)

    # Append the Normal RAG response to chat history
    st.session_state.messages.append({"role": "assistant", "content": normal_rag_response})

    # --- 3. Multi-Agent RAG Response ---
    # Determine the specialized head based on the query content
    if "leave" in query.lower():
        head = "Leave Policy Expert"
    elif "ethics" in query.lower():
        head = "Business Ethics Expert"
    elif "human rights" in query.lower():
        head = "Human Rights Expert"
    else:
        head = "General Policy Expert"

    # Multi-agent prompt construction
    multi_agent_prompt = f'''
    You are an expert in {head}. Provide a detailed response based on the context and your training.
    
    Question: {query}
    
    Context: {context}
    '''
    with st.chat_message("assistant"):
        response_multi = client.chat(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": multi_agent_prompt}],
            stream=False
        )
        multi_agent_response = response_multi['choices'][0]['message']['content']
        st.markdown(multi_agent_response)

    # Append the multi-agent RAG response to chat history
    st.session_state.messages.append({"role": "assistant", "content": multi_agent_response})

    # --- Optional: Check Vagueness and Calculate Relevance Score ---
    # Define vagueness check function
    def check_vagueness(answer):
        vague_phrases = ["I am not sure", "it depends", "vague", "uncertain", "unclear"]
        return any(phrase in answer.lower() for phrase in vague_phrases)

    # Define relevance score calculation
    def calculate_relevance_score(query, response):
        keywords = query.lower().split()
        matches = sum(1 for word in keywords if word in response.lower())
        return matches / len(keywords)

    #finetuned metrics
    is_vague_ft = check_vagueness(fine_tuned_response )
    relevance_score_ft = calculate_relevance_score(query, fine_tuned_response )


    # Normal RAG metrics
    is_vague_normal = check_vagueness(normal_rag_response)
    relevance_score_normal = calculate_relevance_score(query, normal_rag_response)

    # Multi-Agent RAG metrics
    is_vague_multi = check_vagueness(multi_agent_response)
    relevance_score_multi = calculate_relevance_score(query, multi_agent_response)

    # Display vagueness and relevance scores for both RAG responses
    st.markdown(f"**Fine Tuned Vagueness Detected:** {'Yes' if is_vague_ft  else 'No'}")
    st.markdown(f"**Fine Tuned Relevance Score:** {relevance_score_ft:.2f}")
    st.markdown(f"**Normal RAG Vagueness Detected:** {'Yes' if is_vague_normal else 'No'}")
    st.markdown(f"**Normal RAG Relevance Score:** {relevance_score_normal:.2f}")
    st.markdown(f"**Multi-Agent RAG Vagueness Detected:** {'Yes' if is_vague_multi else 'No'}")
    st.markdown(f"**Multi-Agent RAG Relevance Score:** {relevance_score_multi:.2f}")
