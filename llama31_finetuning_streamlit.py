import streamlit as st
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = LlamaForCausalLM.from_pretrained("tecace/llama-3.1-8b-counselor")  # Vessl에서 모델 경로
    tokenizer = LlamaTokenizer.from_pretrained("tecace/llama-3.1-8b-counselor")  # Vessl에서 토크나이저 경로
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit app layout
st.title("Llama 3.1 Fine-tuning Inference")
input_text = st.text_area("Enter your text input here:", height=150)

if st.button("Generate Response"):
    if input_text:
        with st.spinner("Generating response..."):
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"], max_length=150)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success("Generated Response:")
            st.write(generated_text)
    else:
        st.error("Please enter some text to generate a response.")
