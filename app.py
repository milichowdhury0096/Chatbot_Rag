import sys
# Ensure compatibility with SQLite
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai
from langchain_community.vectorstores import Chroma
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Streamlit App Configuration
st.set_page_config(layout="wide")
st.title("Emplochat")

# Sidebar for API Key input
with st.sidebar:
    API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

# Initialize OpenAI client
class OpenAIClient:
    def __init__(self, api_key):
        openai.api_key = api_key

    def chat(self, *args, **kwargs):
        return openai.ChatCompletion.create(*args, **kwargs)

client = OpenAIClient(API_KEY)

persist_directory = '/mount/src/Chatbot_multiagent/embeddings'

# Initialize the Chroma DB client
store = Chroma(persist_directory=persist_directory, collection_name="Capgemini_policy_embeddings")

# Define the embedding function
class OpenAIEmbeddingFunction:
    def __call__(self, texts):
        response = openai.Embedding.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        embeddings = [embedding['embedding'] for embedding in response['data']]
        return embeddings

embed_prompt = OpenAIEmbeddingFunction()

# Define the embedding retrieval function
def retrieve_vector_db(query, n_results=2):
    embedding_vector = embed_prompt([query])[0]
    similar_embeddings = store.similarity_search_by_vector_with_relevance_scores(embedding=embedding_vector, k=n_results)
    results = [embedding for embedding in similar_embeddings]
    return results

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

    # Determine the specialized head based on the query content
    if "leave" in query.lower():
        head = "Leave Policy Expert"
    elif "ethics" in query.lower():
        head = "Business Ethics Expert"
    elif "human rights" in query.lower():
        head = "Human Rights Expert"
    else:
        head = "General Policy Expert"

    prompt = f'''
    You are an expert in {head}. Give a detailed answer based on the context provided and your training.
    
    Question: {query}
    
    Context: {context}
    '''

    st.session_state.messages.append({"role": "user", "content": query})

    # Display the user message
    with st.chat_message("user"):
        st.markdown(query)

    # Generate Normal RAG response
    with st.chat_message("assistant"):
        response = client.chat(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": prompt}],
            stream=False
        )
        normal_response = response['choices'][0]['message']['content']
        st.markdown(normal_response)

    # Append the assistant's Normal RAG response to chat history
    st.session_state.messages.append({"role": "assistant", "content": normal_response})

    # Check for vagueness
    def check_vagueness(answer):
        vague_phrases = ["I am not sure", "it depends", "vague", "uncertain", "unclear"]
        return any(phrase in answer.lower() for phrase in vague_phrases)

    is_vague_normal = check_vagueness(normal_response)

    # Calculate contextual relevance score
    def calculate_contextual_relevance_score(query, response):
        query_embedding = embed_prompt([query])[0]
        response_embedding = embed_prompt([response])[0]
        similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
        return similarity

    contextual_relevance_score_normal = calculate_contextual_relevance_score(query, normal_response)

    # Display Normal RAG vagueness and score metrics
    st.markdown(f"**Normal RAG Vagueness Detected:** {'Yes' if is_vague_normal else 'No'}")
    st.markdown(f"**Normal RAG Contextual Relevance Score:** {contextual_relevance_score_normal:.2f}")

    # Generate Multi-Agent RAG response
    with st.chat_message("assistant"):
        multi_prompt = f'''
        You are an expert in {head}. Provide a detailed response based on the context and your training.
        
        Question: {query}
        
        Context: {context}
        '''
        response_multi = client.chat(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": multi_prompt}],
            stream=False
        )
        multi_response = response_multi['choices'][0]['message']['content']
        st.markdown(multi_response)

    # Append the assistant's Multi-Agent RAG response to chat history
    st.session_state.messages.append({"role": "assistant", "content": multi_response})

    # Check for vagueness in Multi-Agent response
    is_vague_multi = check_vagueness(multi_response)

    # Calculate contextual relevance score for Multi-Agent response
    contextual_relevance_score_multi = calculate_contextual_relevance_score(query, multi_response)

    # Display Multi-Agent RAG vagueness and score metrics
    st.markdown(f"**Multi-Agent RAG Vagueness Detected:** {'Yes' if is_vague_multi else 'No'}")
    st.markdown(f"**Multi-Agent RAG Contextual Relevance Score:** {contextual_relevance_score_multi:.2f}")
