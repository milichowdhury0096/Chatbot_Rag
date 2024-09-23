# Emplochat (with RAG and MultiAgent RAG)

Emplochat is a Streamlit application that allows users to interact with an AI chatbot trained on Capgemini policies. The chatbot utilizes OpenAI's fine-tuned models and a vector database to provide accurate and relevant responses to user queries regarding company policies, ethics, human rights, and more. 

# Link to Streamlit

https://chatbotrag-fwnbyw5vsvq9k5cpc5jauc.streamlit.app/


# Link to Collab files

- https://drive.google.com/file/d/1yZFac3xY7_YxpUXng_vBcarqQz1-fpQe/view?usp=sharing            : EMBEDDINGS

- https://drive.google.com/file/d/1Jd3ri31wiMnW9oh339JfW-AgVF8UJPdi/view?usp=sharing            : RAG      



## Features

- **Chatbot Interface**: Users can easily input their queries in a user-friendly chat interface.
- **Multiple Response Modes**: The chatbot generates responses using different methods:
  
   **Retrieval-Augmented Generation (RAG)**: Emplochat employs RAG techniques to enhance the chatbot's ability to generate contextually relevant responses by retrieving information from a database of Capgemini policies.  
   **Multi-Agent RAG**: The application also incorporates a multi-agent RAG approach, allowing the chatbot to utilize specialized agents that focus on different areas of expertise (e.g., leave policies, business ethics) for more precise and informative answers.
   
- ## Scoring System

Responses are evaluated on a scale from 0.0 to 1.0 for the following criteria:

1. **Contextual Alignment**: 
   - **0.0**: No alignment with the context.
   - **0.2**: Very weak alignment.
   - **0.4**: Some alignment.
   - **0.6**: Moderate alignment.
   - **0.8**: Strong alignment.
   - **1.0**: Perfect alignment with the context.

2. **Clarity**:
   - **0.0**: No clarity; very confusing.
   - **0.2**: Very little clarity; hard to understand.
   - **0.4**: Some clarity; partially understandable.
   - **0.6**: Moderately clear; generally easy to follow.
   - **0.8**: Mostly clear; minor ambiguities.
   - **1.0**: Completely clear; easy to understand.

3. **Depth of Insight**:
   - **0.0**: No valuable insight; irrelevant information.
   - **0.2**: Very little insight; superficial response.
   - **0.4**: Some insight; basic information provided.
   - **0.6**: Moderate insight; useful but not detailed.
   - **0.8**: Valuable insight; adds depth to the answer.
   - **1.0**: Very detailed and insightful; highly informative.


**output.jsonl** consist of question and answers created from the policies which is used for creating fine tuned model in OpenAI.


## Requirements

- openai == 0.28
- pysqlite3-binary
- A complete list of required packages and their versions can be found in the [requirements.txt](requirements.txt) file.

## License
[Mit Licence](LICENSE)
