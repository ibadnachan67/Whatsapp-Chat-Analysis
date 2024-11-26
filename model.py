import streamlit as st
from transformers import pipeline

@st.cache_data(show_spinner="Loading Bert pipline")
def load_sematic_analysis_pipeline():
    model_name = 'bert-base-multilingual-uncased'
    bert_model = pipeline('feature-extraction', model=model_name)
    return bert_model

@st.cache_data(show_spinner="Loading Bert pipline")
def load_sentiment_analysis_pipeline():
    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
    return pipeline("sentiment-analysis", model=model_name)