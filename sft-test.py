import os.path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset,IterableDatasetDict


# 从tim复制过来的
PROMPT_DICT = {
    "prompt_input": (
        "Write a response that appropriately completes the request.\n\n"
        "### Request:\n{instruction}\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Write a response that appropriately completes the request.\n\n"
        "### Request:\n{instruction}\n\n### Response:"
    ),
}
IGNORE_IDX = -100
INPUT_MAX = 400
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


#从tim复制过来的

def main():
    checkpoint_path = "./DeepSeek-R1-Distill-Llama-8B/"
    train_file = "./train_data/WMT23_1_20000.json"
    output_dir = "./checkpoints/test/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 加载数据集
    # copy from tim~ project
    data_files = {"train": train_file}
    extension ="json" ## 确定文件扩展名
    raw_datasets = IterableDatasetDict()
    raw_datasets["train"] = load_dataset(
        extension,  # 数据集类型
        data_files=data_files,
        split="train",  ## 指定的加载分割
        use_auth_token=True,
        streaming=True )
    # copy from tim~ project

    # 加载预训练的 Tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    print("tokenizer.model_max_length:{}".format(tokenizer.model_max_length))
    exit(0)

    # 数据预处理：对输入进行 Tokenizer

    def replace_function(example):  ## 处理数据集中的一条数据
        # 从提示模板中获取提示 # 分别是有input的提示和没有input的提示
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        ## 把数据内容插入到提示模板中
        text = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(
            example)

        example["text"] = text.replace("\\n", "\n")  ##替换换行符
        example["labels"] = f"{example['output']}{tokenizer.eos_token}".replace("\\n", "\n")
        return example

    ## 【用datasets的map方法，对每一条数据执行replace function：
    # 把input插入prompt模板中，给output和badouts加上</s>符号【序列结束标记】
    replaced_dataset = dataset.map(replace_function)

    def tokenize_function(examples):  ## todo:修改该方法
        # return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

        tokenize_output = {"input_ids": [], "attention_mask": [], "labels": []}  ## labels是用于计算损失的shift-labels？

        input_token = tokenizer(examples["text"], add_special_tokens=False)  # text列，所有的prompt，包含input和instruction的部分
        output_token = tokenizer(examples["labels"], add_special_tokens=False)  ## 好输出

        # 主要是要计算attention mask？
        for idx, (s, t) in enumerate(zip(input_token["input_ids"], output_token["input_ids"])):
            Len_s = len(s)
            Len_t = len(t)
            att_s = input_token["attention_mask"][idx][:Len_s]
            att_t = output_token["attention_mask"][idx]

            cur_input_ids = s + t
            cur_labels = [IGNORE_IDX for _ in range(Len_s)] + t  ## promt的部分全都是IGNORE_IDX+真实的好输出
            cur_attention_mask = att_s + att_t

            if cur_input_ids[-1] != tokenizer.eos_token_id:  # 如果最后一个token不是eos则加上eos
                cur_input_ids = cur_input_ids + [tokenizer.eos_token_id]
                cur_labels = cur_labels + [tokenizer.eos_token_id]
                cur_attention_mask = cur_attention_mask + [1]

            Len_cur_input = len(cur_input_ids)

            # 一定要检查长度吗？？
            # assert len(cur_input_ids) == block_size, f"Not match input length {len(cur_input_ids)} and block_size {block_size}, Len_s {Len_s}, Len_t {Len_t}"

        return tokenize_output

    tokenized_dataset = replaced_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["input", "output", "instruction"])

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,  # 保存模型输出的目录
        evaluation_strategy="steps",  # 触发评估的间隔方式
        eval_steps=500,  # 每隔多少步评估验证集
        save_total_limit=1,  # 最多保留最近的两个模型文件
        per_device_train_batch_size=4,  # 每个设备训练时的 batch 大小
        per_device_eval_batch_size=4,  # 每个设备评估时的 batch 大小
        logging_dir="./logs",  # 日志文件目录
        logging_steps=200,  # 多少步打印一次日志
        num_train_epochs=1,  # 总训练轮数
        lr_scheduler_type="linear",  # 学习率调度方法
        learning_rate=5e-5,  # 学习率
        warmup_steps=100,  # 预热步数
        save_steps=500,  # 每隔多少步保存模型
        weight_decay=0.01,  # 权重衰减
        fp16=True,  # 如果支持，则启用混合精度训练
        push_to_hub=False,  # 是否上传到 Hugging Face 共享模型库
        gradient_accumulation_steps=1,  # 梯度累积步数
    )

    # 定义 Hugging Face Trainer 类
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,  # 必须传入 Tokenizer 以进行处理
    )

    # 开始训练 (微调模型)
    trainer.train()

    # 保存训练完成后的模型
    trainer.save_model("./fine_tuned_model")
    print("Model successfully saved!")

    # 评估模型
    results = trainer.evaluate()
    print("Evaluation results:", results)


if __name__ == "__main__":
    main()
