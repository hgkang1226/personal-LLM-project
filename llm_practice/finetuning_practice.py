import torch
from transformers import LlamaForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
import random
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import re
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load pretrained model
tokenizer = AutoTokenizer.from_pretrained("daily_tokenizer_0612")
model = LlamaForCausalLM.from_pretrained('daily_llama_0612')
model.to(device)

# load dataset
data = 'GonzaloA/fake_news'
dataset_fake = load_dataset(data)

dataset_fake = DatasetDict({'train': dataset_fake['train'], 'test': dataset_fake['test']})

data = 'heegyu/news-category-balanced-top10'

dataset_cate = load_dataset(data)

categories = dataset_cate['train'].to_pandas().category.unique().tolist()
categories.sort()
categories = categories[:4]

dataset_cate = dataset_cate.filter(lambda element: element['category'] in categories)

int2label_fake = {0: 'False', 1: 'True'}
label2int_fake = {'False': 0, 'True': 1}

categories = [x.split(' ')[0].lower() for x in categories]
int2label_cate = {i: categories[i] for i in range(len(categories))}
label2int_cate = {int2label_cate[key]:key for key in int2label_cate}

def gen_label(element):
    category = element['category'].split(' ')[0].lower()
    return {'label': label2int_cate[category], 'category': category}

dataset_cate = dataset_cate.map(gen_label)
dataset_cate = dataset_cate['train'].train_test_split(test_size=0.1)


prompt_format1_fake = """Determine if the given article is fake. article: %s  answer: %s"""
prompt_format2_fake = """Is this article fake? article: %s answer: %s"""
prompt_format3_fake = """Return True if the given article is fake. article: %s answer: %s"""

prompts_fake = [prompt_format1_fake, prompt_format2_fake, prompt_format3_fake]
def gen_prompt_fake(element):
    prompt_format = prompts_fake[random.randint(0, len(prompts_fake)-1)]
    return DatasetDict({'input': prompt_format%(element['title'], int2label_fake[element['label']])})


prompt_format1_cate = """Given the article, what is the topic of the article? article: %s  answer: %s"""
prompt_format2_cate = """Determine the topic of the news article. article: %s answer: %s"""
prompt_format3_cate = """What is this article about? business/entertainment/food/healthy/parenting article: %s answer: %s"""

prompts_cate = [prompt_format1_cate, prompt_format2_cate, prompt_format3_cate]

def gen_prompt_cate(element):
    prompt_format = prompts_cate[random.randint(0, len(prompts_cate)-1)]
    return DatasetDict({'input': prompt_format%(element['headline'], int2label_cate[element['label']])})


train_fake = dataset_fake['train'].map(gen_prompt_fake, remove_columns=dataset_fake['train'].column_names)
train_cate = dataset_cate['train'].map(gen_prompt_cate, remove_columns=dataset_cate['train'].column_names)

train_dataset = concatenate_datasets([train_fake, train_cate]).shuffle()

def tokenize(element):
    tokenizer.pad_token = tokenizer.eos_token
    outputs = tokenizer(
        element['input'],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=False,
        return_length=True,
        padding=True
    )

    return {"input_ids": outputs["input_ids"]}


context_length=128
tokenized_datasets = train_dataset.map(
    tokenize, batched=True, remove_columns=train_dataset.column_names
)

# training


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

out = data_collator([tokenized_datasets[i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")


args = TrainingArguments(
    output_dir="combined_instruct_llama",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=1_000,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

trainer.train()

# evaluation

tokenizer = AutoTokenizer.from_pretrained("daily_tokenizer_0612", padding_side='left')


prompt_format1 = """Determine if the given article is fake. article: %s  answer:"""
prompt_format2 = """Is this article fake? article: %s answer:"""
prompt_format3 = """Return True if the given article is fake. article: %s answer:"""

prompts = [prompt_format1, prompt_format2, prompt_format3]

def gen_valid_prompt_fake(element):
    prompt_format = prompts[random.randint(0, len(prompts)-1)]
    return DatasetDict({'input': prompt_format%(element['title'])})


valid_dataset = dataset_fake['test'].select(range(100)).map(gen_valid_prompt_fake)

context_length=128
valid_dataset = valid_dataset.map(
    tokenize, batched=True, remove_columns=['text', 'input', 'Unnamed: 0', 'title']
)



batch_size=4
val_ds = valid_dataset
val_ds.set_format(type='torch')
val_dl = DataLoader(val_ds, batch_size=batch_size)



def acc(pred,label):
    return torch.sum(torch.tensor(pred) == label.squeeze()).item()

model.eval()
val_losses = []
val_acc = 0

for step, batch in enumerate(tqdm(val_dl)):
    label = batch['label']
    input_id= batch['input_ids'].to(device)

    pred = model.generate(input_id, max_length=128)
    decoded_pred = tokenizer.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decoded_pred = [re.findall("answer: (True|False)", x)[0] if re.findall("answer: (True|False)", x) else 'none' for x in decoded_pred]
    decoded_pred = [label2int_fake[x] if x in label2int_fake else -1 for x in decoded_pred]

    val_acc += acc(decoded_pred, label)
    
    if step == 100:
        break

print("val acc: ", val_acc/((step+1)*batch_size))

tokenizer = AutoTokenizer.from_pretrained("daily_tokenizer_0612", padding_side='left')
prompt_format1 = """Given the article, what is the topic of the article? article: %s  answer:"""
prompt_format2 = """Determine the topic of the news article. article: %s answer:"""
prompt_format3 = """What is this article about? business/entertainment/food/healthy/parenting article: %s answer:"""

prompts = [prompt_format1, prompt_format2, prompt_format3]

def gen_valid_prompt_cate(element):
    prompt_format = prompts[random.randint(0, len(prompts)-1)]
    return DatasetDict({'input': prompt_format%(element['headline'])})

valid_dataset = dataset_cate['test'].map(gen_valid_prompt_cate)

context_length=128
valid_dataset = valid_dataset.map(
    tokenize, batched=True, remove_columns=['link', 'headline', 'category', 'short_description', 'authors', 'date', 'input']
)


batch_size=4
val_ds = valid_dataset.select(range(100))
val_ds.set_format(type='torch')
val_dl = DataLoader(val_ds, batch_size=batch_size)



model.eval()
val_losses = []
val_acc = 0

for step, batch in enumerate(tqdm(val_dl)):
    device='cuda:0'
    label = batch['label']
    input_id= batch['input_ids'].to(device)

    pred = model.generate(input_id, max_length=150)
    decoded_pred = tokenizer.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decoded_pred = [re.findall("answer: ([a-z]+)", x)[0] if re.findall("answer: ([a-z]+)", x) else 'none' for x in decoded_pred]
    decoded_pred = [label2int_cate[x] if x in label2int_cate else -1 for x in decoded_pred]

    val_acc += acc(decoded_pred, label)

print("val acc: ", val_acc/len(val_dl.dataset))

# save model
model.save_pretrained('llama_combined_0618')