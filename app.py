import sys
# Ensure compatibility with SQLite
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

persist_directory = '/mount/src/Chatbot_Rag/embeddings'

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

    # Generate Fine-tuned Model response
    with st.chat_message("assistant"):
        fine_tuned_prompt = f'''
        You are an expert in {head}. Provide a detailed response based on the context and your training.
        
        Question: {query}
        
        Context: {context}
        '''
        response_fine_tuned = client.chat(
            model="ft:gpt-3.5-turbo-0125:personal::A9eKNr3q",  # Replace with your fine-tuned model ID
            messages=[{"role": "system", "content": fine_tuned_prompt}],
            stream=False
        )
        fine_tuned_response = response_fine_tuned['choices'][0]['message']['content']
        st.markdown(fine_tuned_response)

    # Append the assistant's Fine-tuned response to chat history
    st.session_state.messages.append({"role": "assistant", "content": fine_tuned_response})

    # Check for vagueness in Fine-tuned response
    is_vague_fine_tuned = check_vagueness(fine_tuned_response)
    relevance_score_fine_tuned = calculate_relevance_score(query, fine_tuned_response)

    # Display Fine-tuned RAG vagueness and score metrics
    st.markdown(f"**Fine-tuned RAG Vagueness Detected:** {'Yes' if is_vague_fine_tuned else 'No'}")
    st.markdown(f"**Fine-tuned RAG Relevance Score:** {relevance_score_fine_tuned:.2f}")

    # Generate Normal RAG response
    with st.chat_message("assistant"):
        normal_prompt = f'''
        You are an expert in {head}. Give a detailed answer based on the context provided and your training.
        
        Question: {query}
        
        Context: {context}
        '''
        response_normal = client.chat(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": normal_prompt}],
            stream=False
        )
        normal_response = response_normal['choices'][0]['message']['content']
        st.markdown(normal_response)

    # Append the assistant's Normal RAG response to chat history
    st.session_state.messages.append({"role": "assistant", "content": normal_response})

    # Check for vagueness in Normal response
    is_vague_normal = check_vagueness(normal_response)
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

    # Compare responses
    st.markdown("### Comparison of Responses:")
    st.markdown(f"**Fine-tuned RAG Response:** {fine_tuned_response}")
    st.markdown(f"**Normal RAG Response:** {normal_response}")
    st.markdown(f"**Multi-Agent RAG Response:** {multi_response}")
