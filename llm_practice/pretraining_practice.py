from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import LlamaConfig
from transformers import LlamaForCausalLM
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer


# Load dataset

raw_dataset = load_dataset('ccdv/cnn_dailymail', '3.0.0')

raw_dataset['train'][0]['article'][:200]

sampled_dataset = DatasetDict(
    {
        "train": raw_dataset['train'].select(range(50000)).shuffle(),
        "valid": raw_dataset['test'].select(range(5000)).shuffle()
    }
)


# Tokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

def get_training_corpus(ds):
    return(
        ds[i:i+1000]['article'] for i in range(0, len(ds), 1000)
    )


training_corpus = get_training_corpus(raw_dataset['train'])
tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size=50257)

sample_text = "It's official: U.S. President Barack Obama wants lawmakers to weigh in on whether to use military force in Syria"

tokenizer.tokenize(sample_text)

tokenizer(sample_text, return_length=True)

context_length = 128

def tokenize(batch):
    outputs = tokenizer(
        batch['article'],
        max_length=context_length,
        truncation=True,
        return_overflowing_tokens=True,
        return_length=True
    )
    
    input_batch = []
    for length, input_ids in zip(outputs['length'], outputs['input_ids']):
        if length==context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = sampled_dataset.map(tokenize, batched=True, remove_columns=raw_dataset['train'].column_names)


# Load model

configuration = LlamaConfig()

configuration = LlamaConfig(**{
  "bos_token_id": 50256,
  "eos_token_id": 50256,
  "hidden_act": "silu",
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1376,
  "max_position_embeddings": 128,
  "model_type": "llama",
  "num_attention_heads": 4,
  "num_hidden_layers": 4,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": False,
  "transformers_version": "4.28.0",
  "use_cache": True,
  "vocab_size": 50257
})


model = LlamaForCausalLM(configuration)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

prompt = "It's official: U.S. President Barack Obama wants lawmakers to weigh in on whether to use military force in "

inputs = tokenizer(prompt, return_tensors='pt')
inputs.to(device)

generate_ids = model.generate(inputs.input_ids, max_length=50)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# Training

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

out = data_collator([tokenized_datasets['train'][i] for i in range(3)])

for key in out:
    print(f"{key}: {out[key].shape}")



batch_size = 32
logging_steps = 1000
learning_rate=5e-4
num_epochs=1

args = TrainingArguments(
    output_dir='newsllama',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='steps',
    eval_steps=logging_steps,
    logging_steps=logging_steps,
    save_steps=logging_steps,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=logging_steps,
    lr_scheduler_type='cosine',
    learning_rate=5e-4,
    fp16=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid']
)

trainer.train()

# evaluation

prompt = "It's official: U.S. President Barack Obama wants lawmakers to weigh in on whether to use military force in "

inputs = tokenizer(prompt, return_tensors='pt')
inputs.to(device)

generate_ids = model.generate(inputs.input_ids, max_length=50)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

prompt = """It's official: U.S. President Barack Obama wants lawmakers to weigh in on whether to use military force in"""
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to("cuda:0")


generate_ids = model.generate(inputs.input_ids, max_length=50)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

prompt = """Shall I compare thee to a summer’s day?
Thou art more lovely and more temperate:
Rough winds do shake the"""
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to("cuda:0")


generate_ids = model.generate(inputs.input_ids, max_length=50)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

prompt = """As a scientific endeavor, machine learning grew out of the quest for artificial intelligence (AI). In the early days of AI as an academic discipline, some researchers were interested in"""
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to("cuda:0")


generate_ids = model.generate(inputs.input_ids, max_length=50)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
