import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset


'''
    模型下载
    
    
    from modelscope import snapshot_download
    model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')
    or
    git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B-Instruct.git
    
    ref:https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files
'''


def process_func(example):
    MAX_LENGTH = 1024  # 最大输入长度
    output_group_length = 128  # 响应内容的最大长度（可根据实际需求调整）
    input_group_length = MAX_LENGTH - output_group_length  # 用户内容的剩余长度

    p = '输入反应式，输出反应类别[1-10]，共10种。\n不需要给出具体过程，直接给出答案。\nexample1：\ninput：IC[I:10].N[c:9]1[s:8][cH:7][c:6]([C:4]([O:3][CH2:2][CH3:1])=[O:5])[n:11]1>>[CH3:1][CH2:2][O:3][C:4](=[O:5])[c:6]1[cH:7][s:8][c:9]([I:10])[n:11]1\noutput：9\nexample2：\ninput：[N:1]#[C:2][CH:3]1[CH2:4][CH:5]1[C:6](=[O:7])[c:8]1[cH:9][cH:10][c:11]([Cl:12])[cH:13][c:14]1[F:15]>>[N:1]#[C:2][CH:3]1[CH2:4][CH:5]1[CH:6]([OH:7])[c:8]1[cH:9][cH:10][c:11]([Cl:12])[cH:13][c:14]1[F:15]\noutput：7'

    instruction_input = example['rxn_smiles']
    response_output = str(example['class'])
    instruction = tokenizer(
        "<|im_start|>system\n"+p+"<|im_end|>\n<|im_start|>user\n" + instruction_input + "<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
        truncation=True,
        max_length=input_group_length
    )

    response = tokenizer(
        response_output + "<|im_end|>",
        add_special_tokens=False,
        truncation=True,
        max_length=output_group_length
    )

    input_ids = instruction['input_ids'] + response['input_ids'] + [tokenizer.pad_token_id]
    attention_mask = instruction['attention_mask'] + response['attention_mask'] + [1]
    labels = [-100] * len(instruction['input_ids']) + response['input_ids'] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


checkpoint_save_path = datetime.datetime.now().strftime("%Y%m%d%H")

print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained('pretrain_model/Qwen2___5-0___5B-Instruct', use_fast=False,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print('Loading model')
model = AutoModelForCausalLM.from_pretrained('pretrain_model/Qwen2___5-0___5B-Instruct', device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()

# 将JSON文件转换为CSV文件
print('Loading dataset')
df = pd.read_csv('data/50k-all.csv')
ds = Dataset.from_pandas(df)
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

training_args = TrainingArguments(
    output_dir="50k_model/" + checkpoint_save_path,  # 模型保存路径
    learning_rate=5e-5, # 原本是5e-7
    lr_scheduler_type="cosine",
    logging_steps=100,  # 窗口打印日志
    max_steps=200000,  # 设置训练的step，根据实际情况修改
    # save_stategy="steps",
    save_steps=10000,  # 每1000个step保存一次
    gradient_accumulation_steps=1,  # 梯度累加策略,越大显存需求越少，训练速度越慢
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    remove_unused_columns=False,
    report_to=["tensorboard"]
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),    # 如果删掉这个，就需要手动填充至最大token长度
)

# 开始训练
trainer.train()
# 保存模型
trainer.save_model(training_args.output_dir)