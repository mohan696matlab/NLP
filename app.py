import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_models(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",  # Normalized float 4-bit (recommended)
        bnb_4bit_compute_dtype=torch.float16,  
        bnb_4bit_use_double_quant=True  # Improves performance by applying second quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = 128001
    return model, tokenizer

@st.cache_resource
def load_embedding_model(embedding_model_name="all-MiniLM-L12-v2"):
    embedding_model = SentenceTransformer(embedding_model_name)
    return embedding_model

@st.cache_data
def load_data():
    with open("cleaned_text.pkl", "rb") as f:
        cleaned_text = pickle.load(f)
    embeddings = torch.load("vector_embeddings.pt").to(device)
    return cleaned_text, embeddings

model, tokenizer = load_models()
embedding_model = load_embedding_model()
cleaned_text, embeddings = load_data()

def RAG_GITA(query):
    query_encoded = embedding_model.encode([query], convert_to_tensor=True)
    similarities = embedding_model.similarity(query_encoded, embeddings)
    scores, top_5_indices = torch.topk(similarities[0], k=6)

    CONTEXT_TEXT = '\n'.join([cleaned_text[idx] for idx in top_5_indices if similarities[0][idx] > 0])

    conversation = [
        {"role": "user", "content": f'''{query}
    You are a compassionate guide. You answer questions based only on the given context of Gita and do not add any extra information. 
    Quote the text you have used from the Gita context
    If the question asked by the user is not raleted to the context, you say 'this is not relate dto bhagabat gita'. 
    Your goal is to first interprete the writtings of gita in the given context and then try your best to relate to user's query.
    Answer in simple english that a 15 year old can undrestand using your interpretation of the given text. 
    Gita CONTEXT : {CONTEXT_TEXT}
    Answer should not be longer than 200 words. Do not use markdown format. Quote the text you have used from the Gita context'''},
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=256
        )


    processed_text = tokenizer.decode(output[0][len(inputs.input_ids[0]) + 3:], skip_special_tokens=True)
    return processed_text,CONTEXT_TEXT

st.title("Gita Question Answering")

query = st.text_area("Ask your question about the Gita:", "Is there an afterlife? What happens when we die?")

if st.button("Get Answer") and query:
    with st.spinner("Fetching answer..."):
        response, CONTEXT_TEXT = RAG_GITA(query)
    st.write(response)
    with st.expander("See Context from Gita"):
        st.markdown(
            f'<div style="background-color:#5c4605; padding:10px; border-radius:5px;">{CONTEXT_TEXT}</div>', 
            unsafe_allow_html=True
        )