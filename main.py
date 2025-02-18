# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load model directly

    checkpoint_path="./DeepSeek-R1-Distill-Llama-8B/"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)


    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    pipe = pipeline("text-generation", model=model,tokenizer=tokenizer)

    print(pipe(messages))