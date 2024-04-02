from fastapi import FastAPI
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import re
# uvicorn backend:app --reload

# Initialize instance of FastAPI
app = FastAPI()



# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Birchlabs/llama-13b-stepwise-tokenizer")
# jodiambra/llama-2-7b-finetuned-python-qa_tokenizer

# load model
model = LlamaForCausalLM.from_pretrained('princeton-nlp/Sheared-LLaMA-1.3B')

# h2oai/h2ogpt-4096-llama2-7b-chat
# h2oai/h2ogpt-4096-llama2-13b-chat
# h2oai/h2ogpt-4096-llama2-7b-chat
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# in case of multi GPUs
# if torch.cuda.is_available():
#     n_gpus = torch.cuda.device_count()
#     if n_gpus > 1:
#         model = torch.nn.DataParallel(model, device_ids=list(range(n_gpus)))

model.to(device)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(device)
    
    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=128, 
                                  num_beams=1, no_repeat_ngram_size=3, early_stopping=True)
    
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    output = re.sub(r'^.*?[.?!]', '', output)
    
    if not output.endswith(('!', '?', '.')):
        output = re.sub(r'[^.!?]*$', '', output)
    
    return output.strip()

# def get_answer(prompt):
    
#     inputs = tokenizer(prompt, return_tensors="pt")
#     inputs.to(device)

#     # Generate
#     generate_ids = model.generate(inputs.input_ids, max_length=inputs.input_ids.shape[1]+5)
#     # generate_ids = model.generate(inputs.input_ids, max_length=256)
    
#     output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
#     output = output[len(prompt):]

#     return output

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
 
