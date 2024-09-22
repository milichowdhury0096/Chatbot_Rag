- https://drive.google.com/file/d/1yZFac3xY7_YxpUXng_vBcarqQz1-fpQe/view?usp=sharing

- https://drive.google.com/file/d/1Jd3ri31wiMnW9oh339JfW-AgVF8UJPdi/view?usp=sharing


# Emplochat

Emplochat is a Streamlit application that allows users to interact with an AI chatbot trained on Capgemini policies. The chatbot utilizes OpenAI's fine-tuned models and a vector database to provide accurate and relevant responses to user queries regarding company policies, ethics, human rights, and more. 

## Features

- **Chatbot Interface**: Users can easily input their queries in a user-friendly chat interface.
- **Multiple Response Modes**: The chatbot generates responses using different methods:
  - **Normal RAG**: A general-purpose response generation using a pre-trained model.
  - **Multi-Agent RAG**: Incorporates multiple expert responses for enhanced accuracy.
- **Contextual Responses**: Generates responses based on user queries and relevant policy documents.
- **Vagueness Detection**: The chatbot checks for vague responses and informs the user accordingly.

output.jsonl consist of question and answers created from the policies which is used for creating fine tuned model in OpenAI.


## Requirements

- Python 3.8 or higher
- openai == 0.28
- pysqlite3-binary

## Link to Streamlit

https://chatbotrag-fwnbyw5vsvq9k5cpc5jauc.streamlit.app/

## License
Mit Licence
