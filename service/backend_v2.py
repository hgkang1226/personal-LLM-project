from fastapi import FastAPI
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

app = FastAPI()

tokenizer = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("Birchlabs/llama-13b-stepwise-tokenizer")
    model = LlamaForCausalLM.from_pretrained('h2oai/h2ogpt-4096-llama2-7b-chat')
    model.to(device)

def get_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(device)
    generate_ids = model.generate(inputs.input_ids, max_length=inputs.input_ids.shape[1]+5)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    output = output[len(prompt):]
    return output

@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.get('/chat_test')
async def test(user_message):
    return {'message': get_answer(user_message)}

@app.post('/chat')
async def chat(param: dict={}):
    user_message = param.get('user_message', ' ')
    return {'message': get_answer(user_message)}