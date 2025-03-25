from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

app = FastAPI()

# Load the model and tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4",  # Normalized float 4-bit (recommended)
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_use_double_quant=True  # Improves performance by applying second quantization
)
model =  AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    # torch_dtype=torch.bfloat16,
    device_map=device,
)

tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = 128001

# Define the Pydantic model for the request body
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")  # Define a route for the root URL
async def read_root():
    return {"message": "Welcome to the LLaMA chatbot API!"}
@app.post("/generate/")
async def generate_text(prompt: PromptRequest):

    inputs = tokenizer(prompt.prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    return {"response": tokenizer.decode(output[0][len(inputs.input_ids[0])+3:], skip_special_tokens=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)