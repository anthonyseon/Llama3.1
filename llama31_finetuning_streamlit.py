import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("tecace/llama-3.1-8b-counselor/1")
    tokenizer = AutoTokenizer.from_pretrained("tecace/llama-3.1-8b-counselor/1")
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app title
st.title("LLaMA 3.1 Text Generation")

# Text input from the user
input_text = st.text_area("Enter your text here:")

if st.button("Generate"):
    if input_text:
        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Generate model output
        outputs = model.generate(inputs["input_ids"], max_length=100)
        
        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the result
        st.write("Generated Text:")
        st.write(generated_text)
    else:
        st.write("Please enter some text to generate a response.")