import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


def main():
    # 确保使用 GPU（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 模型和数据集定义
    model_name = "gpt2"  # 替换为所需模型（如 "gpt2" 或其他预训练 LLM）
    dataset_name = "wikitext"  # 示例数据集
    # 加载数据集，用作 SFT 数据
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1")

    # 加载预训练的 Tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # 确保 GPT 模型有一个正确的 pad_token
    model = AutoModelForCausalLM.from_pretrained(model_name)



    # 数据预处理：对输入进行 Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="./results",  # 保存模型输出的目录
        evaluation_strategy="steps",  # 触发评估的间隔方式
        eval_steps=500,  # 每隔多少步评估验证集
        save_total_limit=2,  # 最多保留最近的两个模型文件
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
