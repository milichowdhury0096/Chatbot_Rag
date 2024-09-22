import sys
# Ensure compatibility with SQLite
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai
from langchain_community.vectorstores import Chroma
import numpy as np
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

    # Score the response
    def calculate_relevance_score(query, response):
        keywords = query.lower().split()
        matches = sum(1 for word in keywords if word in response.lower())
        return matches / len(keywords)

    relevance_score_normal = calculate_relevance_score(query, normal_response)

    # Display Normal RAG vagueness and score metrics
    st.markdown(f"**Normal RAG Vagueness Detected:** {'Yes' if is_vague_normal else 'No'}")
    st.markdown(f"**Normal RAG Relevance Score:** {relevance_score_normal:.2f}")

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
    relevance_score_multi = calculate_relevance_score(query, multi_response)

    # Display Multi-Agent RAG vagueness and score metrics
    st.markdown(f"**Multi-Agent RAG Vagueness Detected:** {'Yes' if is_vague_multi else 'No'}")
    st.markdown(f"**Multi-Agent RAG Relevance Score:** {relevance_score_multi:.2f}")

    # Final scoring function
    def calculate_clarity_score(response):
        # Simple clarity check based on length and complexity
        if len(response.split()) < 10:
            return 0.5  # Low score for too short responses
        elif any(word in response.lower() for word in ["complex", "difficult"]):
            return 0.5  # Low score for complex responses
        return 1.0  # High score for clear responses

    def check_factuality(retrieved_context, response):
        # Check if any context is directly referenced in the response
        context_keywords = set(retrieved_context.lower().split())
        response_keywords = set(response.lower().split())
        if context_keywords.intersection(response_keywords):
            return 1.0  # Factual if it references the context
        return 0.0  # Not factual

    def check_context_appropriateness(retrieved_context, response):
        # Basic check for relevance to the retrieved context
        if any(word in response.lower() for word in retrieved_context.lower().split()):
            return 1.0  # Contextually appropriate
        return 0.0  # Not appropriate

    def calculate_final_score(query, response, retrieved_context):
        relevance = calculate_relevance_score(query, response)
        clarity = calculate_clarity_score(response)
        vagueness_penalty = 0 if not check_vagueness(response) else -0.1
        factuality = check_factuality(retrieved_context, response)
        contextual_appropriateness = check_context_appropriateness(retrieved_context, response)

        # Weigh and combine the scores
        final_score = (relevance * 0.4) + (factuality * 0.3) + (clarity * 0.2) + vagueness_penalty + (contextual_appropriateness * 0.2)
        return final_score

    # Calculate final scores for both responses
    final_score_normal = calculate_final_score(query, normal_response, context)
    final_score_multi = calculate_final_score(query, multi_response, context)

    st.markdown(f"**Final Score for Normal RAG Response:** {final_score_normal:.2f}")
    st.markdown(f"**Final Score for Multi-Agent RAG Response:** {final_score_multi:.2f}")
