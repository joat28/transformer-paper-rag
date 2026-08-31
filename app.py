import streamlit as st
from retrieval.retrieval import generate_results


st.header("Attention Paper QnA")

question = st.text_input("Enter a question to search:")

if st.button("Get Answer") and question.strip():
    with st.spinner("Reading paper..."):
        response = generate_results(question)
    st.subheader("Results:")
    st.markdown(
        response,
        unsafe_allow_html=False
    )