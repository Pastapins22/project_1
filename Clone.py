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
with st.sidebar:
    # File uploader for Excel files
    uploaded_files = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'], accept_multiple_files=True)

#Coloumn 1
col1, col2 = st.columns(2)

with col1:
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Read the Excel file
                df = pd.read_excel(uploaded_file)
                
                # Display the file name and DataFrame
                st.write(f"**Filename:** {uploaded_file.name}")
                st.dataframe(df)
            except Exception as e:
                st.error(f"An error occurred while processing {uploaded_file.name}: {e}")

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





