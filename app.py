import streamlit as st
from indexing.indexing import generate_terms_and_jokes

st.header("Comedy Generator AI")
comedy_term = st.text_input("Enter something")

if st.button("Generate Terms and Joke on selected term") and comedy_term != None :
    with st.spinner("Generating .. Jokes .. "):
        terms, joke = generate_terms_and_jokes(comedy_term)
    
    st.subheader("Generated Terms")
    st.write(terms)
    st.subheader("Joke")
    st.write(joke)
    

