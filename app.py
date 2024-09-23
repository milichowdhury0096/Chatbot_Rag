import sys
# Ensure compatibility with SQLite
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import openai
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
import os

# Streamlit App Configuration
st.set_page_config(layout="wide")
st.title("Emplochat")

# Sidebar for API Key input
with st.sidebar:
    API_KEY = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

if not API_KEY:
    st.error("Please enter your OpenAI API Key.")
    st.stop()

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

# Precompute embeddings for heads
heads_keywords = {
    "Leave Policy Expert": ["leave", "absence", "vacation", "time off"],
    "Business Ethics Expert": ["ethics", "integrity", "compliance", "conduct"],
    "Human Rights Expert": ["human rights", "equality", "fairness", "justice"],
    "General Policy Expert": ["policy", "guideline", "procedure", "standard"]
}

# Precompute embeddings for the heads
heads_embeddings = {
    head: embed_prompt(keywords) for head, keywords in heads_keywords.items()
}

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

    # Get the embedding for the user's query
    query_embedding = embed_prompt([query])[0]

    # Calculate cosine similarity
    def cosine_similarity(vec_a, vec_b):
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

    # Determine the specialized head based on cosine similarity
    head_scores = {
        head: cosine_similarity(query_embedding, head_embedding[0])
        for head, head_embedding in heads_embeddings.items()
    }

    # Select the head with the highest score
    head = max(head_scores, key=head_scores.get)

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

    # Function to assess quality of responses
    def assess_quality(response):
        assessment_prompt = f"""
        Please evaluate the following response based on its effectiveness in addressing the user's query. Rate it on a scale of 0 to 1 for each of the following criteria:

        1. **Contextual Alignment**: How well does the response relate to the context provided? (0 = no alignment, 1 = perfect alignment)
        2. **Clarity**: Is the response easy to understand and free from ambiguity? (0 = very unclear, 1 = completely clear)
        3. **Depth of Insight**: Does the response provide valuable information or insights? (0 = no insight, 1 = very detailed and insightful)
        4. **Engagement**: Does the response encourage further dialogue or exploration of the topic? (0 = no engagement, 1 = highly engaging)

        Provide a score between 0 and 1 for each criterion, followed by an overall score (0 to 1) indicating the overall quality of the response.

        Response: "{response}"

        Format your answer as follows:
        - Contextual Alignment: [score]
        - Clarity: [score]
        - Depth of Insight: [score]
        - Engagement: [score]
        - Overall Score: [score]
        """
        
        assessment_response = client.chat(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": assessment_prompt}]
        )
        
        score_lines = assessment_response['choices'][0]['message']['content'].strip().split('\n')
        
        scores = {}
        for line in score_lines:
            key, value = line.split(': ')
            scores[key.strip()] = float(value) if value.replace('.', '', 1).isdigit() else 0.0

        return scores

    # Assess the quality of the Normal RAG response
    normal_quality_scores = assess_quality(normal_response)

    # Display Normal RAG quality scores
    st.markdown(f"**Normal RAG Quality Scores:**")
    st.markdown(f"- Contextual Alignment: {normal_quality_scores['Contextual Alignment']:.2f}")
    st.markdown(f"- Clarity: {normal_quality_scores['Clarity']:.2f}")
    st.markdown(f"- Depth of Insight: {normal_quality_scores['Depth of Insight']:.2f}")
    st.markdown(f"- Engagement: {normal_quality_scores['Engagement']:.2f}")
    st.markdown(f"- Overall Score: {normal_quality_scores['Overall Score']:.2f}")

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

    # Assess the quality of the Multi-Agent RAG response
    multi_quality_scores = assess_quality(multi_response)

    # Display Multi-Agent RAG quality scores
    st.markdown(f"**Multi-Agent RAG Quality Scores:**")
    st.markdown(f"- Contextual Alignment: {multi_quality_scores['Contextual Alignment']:.2f}")
    st.markdown(f"- Clarity: {multi_quality_scores['Clarity']:.2f}")
    st.markdown(f"- Depth of Insight: {multi_quality_scores['Depth of Insight']:.2f}")
    st.markdown(f"- Engagement: {multi_quality_scores['Engagement']:.2f}")
    st.markdown(f"- Overall Score: {multi_quality_scores['Overall Score']:.2f}")
