import streamlit as st
from helper import translate_to_FRENCH

# Streamlit app
def main():
    st.title("Translate English to French with LSTMs")
    st.write("Enter an English sentence below (max 100 characters):")
    
    # Input text box for user input
    english_input = st.text_input("Enter English text:")
    
    # Button to trigger translation
    if st.button("Translate"):
        if len(english_input) > 100:
            st.error("Maximum length of input sentence is 100 characters.")
        elif len(english_input) == 0:
            st.warning("Please enter a sentence.")
        else:
            # Translate the input sentence
            french_translation = translate_to_FRENCH(english_input)
            st.success("French Translation:")
            st.write(french_translation)

if __name__ == "__main__":
    main()
