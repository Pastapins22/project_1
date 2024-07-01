import streamlit as st
import pandas as pd
import emoji
from streamlit_lottie import st_lottie
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import requests

#App Config
st.set_page_config(page_title="Agent Assist", page_icon="🤖")

#Sidebar Essentials which includes a file uploader and an API key uploader
# st.title("Main Page")
with st.sidebar:
    # st.title("Hey I am  your Assistant")
    image = st.file_uploader("Please upload your Document", type=( "pdf", "doc", "xls"))
    if image is not None:
        st.image(image)

    open_api_key = st.text_input("Open API Key", key='chatbot_api_key', type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

#Coloumn 1
col1, col2 = st.columns(2)

with col1:
    # Loading CSV file
    df = pd.read_csv('consumer_data.csv')  # Ensure this path is correct

    #Display csv file
    st.subheader("Consumer Data")
    st.write(df)

    # Loading pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Adding a padding token to the tokenizer because GPT-2 doesn't have one by default
    tokenizer.pad_token = tokenizer.eos_token

    # Function to generate a response using GPT-2 with attention mask and padding
    def generate_response(prompt):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = (inputs != tokenizer.pad_token_id).long()
    
        outputs = model.generate(
            inputs, 
            max_length=150, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2, 
            early_stopping=True,
            num_beams=5,  # Using beam search with more than 1 beam
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    # Function to search for information in the CSV file
    def search_csv(query):
        # Simple search implementation
        results = df[df.apply(lambda row: query.lower() in row.to_string().lower(), axis=1)]
        if not results.empty:
            return results.to_string(index=False)
        else:
            return "Sorry, I couldn't find any information related to your query in the CSV file."

    # Function to determine if the query is related to the CSV file
    def is_csv_related_query(query):
        csv_keywords = ["column", "demand", "data"]  # Add more keywords relevant to your CSV
        return any(keyword in query.lower() for keyword in csv_keywords)

    # Streamlit app
    def main():
        # st.title("")

        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        user_input = st.text_input("You:", "")
        if st.button("Send"):
            if user_input.lower() in ["exit", "quit", "bye"]:
                st.session_state.conversation.append("Chatbot: Goodbye!")
            else:
                # Check if the query is related to the CSV file
                if is_csv_related_query(user_input):
                    response = search_csv(user_input)
                else:
                    response = generate_response(user_input)
            
                st.session_state.conversation.append(f"You: {user_input}")
                st.session_state.conversation.append(f"Chatbot: {response}")

        for chat in st.session_state.conversation:
            st.write(chat)

    if __name__ == "__main__":
        main()


with col2:
    def load_lottieflies(filepath:str):
        with open(filepath,"r") as f:
            return json.load(f)

    lottie_coding = load_lottieflies("animation.json")
    st.write("### Alex here!\nYour Agent Assitance")
    st_lottie(
        lottie_coding,
        speed = 1,
        reverse= False,
        loop=True,
        quality="medium",#High or Low
        height =300,
        width=300,
        key=None,

    )
    st.caption("Recommended prompts")

        

