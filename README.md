# Link to Collab files

- https://drive.google.com/file/d/1yZFac3xY7_YxpUXng_vBcarqQz1-fpQe/view?usp=sharing

- https://drive.google.com/file/d/1Jd3ri31wiMnW9oh339JfW-AgVF8UJPdi/view?usp=sharing


# Emplochat

Emplochat is a Streamlit application that allows users to interact with an AI chatbot trained on Capgemini policies. The chatbot utilizes OpenAI's fine-tuned models and a vector database to provide accurate and relevant responses to user queries regarding company policies, ethics, human rights, and more. 

## Features

- **Chatbot Interface**: Users can easily input their queries in a user-friendly chat interface.
- **Multiple Response Modes**: The chatbot generates responses using different methods:
  - **Normal RAG**: A general-purpose response generation using a pre-trained model.
  - **Multi-Agent RAG**: Incorporates multiple expert responses for enhanced accuracy.
- **Vagueness Detection**: The chatbot checks for vague responses and informs the user accordingly.
- ## Comprehensiveness Score

The **Comprehensiveness Score** evaluates the extent to which a response covers the information requested in the user's query. This score ranges from 0 to 1, where:

- **0** indicates that the response does not provide any relevant information compared to the query.
- **1** signifies that the response fully meets or exceeds the expected information based on the query.

### Calculation

The Comprehensiveness Score is calculated using the following formula:

\[
\text{Comprehensiveness Score} = \frac{\text{Response Length}}{\text{Maximum Expected Length}}
\]

- **Response Length**: The number of words in the assistant's response.
- **Maximum Expected Length**: A predefined maximum number of words expected in a response (e.g., 200 words). This value can be adjusted based on the expected length of informative responses.

The score is normalized to ensure it remains within the range of 0 to 1, facilitating easy comparison across different responses.

### Interpretation

- A higher Comprehensiveness Score indicates a more thorough and informative response, while a lower score suggests that the response may lack detail or depth regarding the user's query.

**output.jsonl** consist of question and answers created from the policies which is used for creating fine tuned model in OpenAI.


## Requirements

- Python 3.8 or higher
- openai == 0.28
- pysqlite3-binary

## Link to Streamlit

https://chatbotrag-fwnbyw5vsvq9k5cpc5jauc.streamlit.app/

## License
Mit Licence
