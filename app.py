import streamlit as st
import numpy as np
import torch 
from transformers import BertTokenizer , BertForSequenceClassification,TextClassificationPipeline

# @st.cache(allow_output_mutation=True)
@st.cache_data


def get_model():
    tokenizer = BertTokenizer.from_pretrained("Ronysalem/BertCommentsClassifer")
    model = BertForSequenceClassification.from_pretrained("Ronysalem/BertCommentsClassifer")
    return tokenizer, model

tokenizer, model = get_model()

st.title("Bert Comments Classifier")
user_input = st.text_area("Enter Text To Analyze")
button = st.button("Analyze")

if user_input and button:
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512, return_tensors='pt')
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    st.write("Label: ",pipeline(user_input)[0]["label"])
    st.write("score: ",pipeline(user_input)[0]["score"])

