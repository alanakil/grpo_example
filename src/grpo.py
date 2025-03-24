# %%
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import random, numpy as np
import wandb
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# %%
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# %%
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


# %%
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]

    return answer


# %%
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# %%
def get_gsm8k_questions(split="train", num_samples=None) -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    if num_samples:
        data = data.shuffle(seed=42).select(range(num_samples))
    return data


dataset = get_gsm8k_questions("train", None)
validation_dataset = get_gsm8k_questions("test", None)


# %%
def correctness_reward_fn(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print(
    #     "-" * 20,
    #     f"Question:\n{q}",
    #     f"Answer:\n{answer[0]}",
    #     f"\nResponse:\n{responses[0]}",
    #     f"\nExtracted:\n{extracted_responses[0]}",
    # )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


# %%
def int_reward_fn(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# %%
def strict_format_reward_fn(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# %%
def soft_format_reward_fn(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# %%
def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("<answer>\n") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# %%
def xmlcount_reward_fn(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# %%
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

output_dir = "output/Qwen-0.5B-GRPO"
run_name = "Qwen-0.5B-GRPO-gsm8k"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=2,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    # vllm_gpu_memory_utilization=0.3,
    # vllm_device="cuda:0",
    report_to="wandb",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# %%
wandb.init(project="GRPO-gsm8k", name=run_name, config=training_args.to_dict())

# %%
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_fn,
        soft_format_reward_fn,
        strict_format_reward_fn,
        int_reward_fn,
        correctness_reward_fn,
    ],
    args=training_args,
    train_dataset=dataset,
    eval_dataset=validation_dataset,  # Add evaluation dataset
)


# %%
def evaluate_model(model, tokenizer, dataset, sample_size=None):
    model.eval()
    correct = 0
    total = 0
    examples = []

    if sample_size:
        dataset = random.sample(list(dataset), sample_size)

    for item in tqdm(dataset, desc="Evaluating"):
        prompt = tokenizer.apply_chat_template(
            item["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        assistant_marker = "<|im_start|>assistant"
        if assistant_marker not in response_text:
            response_text = ""
        # Take everything after "<|im_start|>assistant"
        response_text = response_text.split(assistant_marker)[-1]
        response_text = response_text.split("<|im_end|>")[0]
        pred_answer = extract_xml_answer(response_text)
        true_answer = item["answer"].strip()

        is_correct = pred_answer == true_answer
        correct += is_correct
        total += 1

        examples.append(
            {
                "question": item["prompt"][-1]["content"],
                "predicted": pred_answer,
                "true": true_answer,
                "correct": is_correct,
            }
        )

    accuracy = correct / total
    return accuracy, examples


# %%
for num_train_samples in [5, 10, 20]:
    print(f"\n{'=' * 20} Training with {num_train_samples} samples {'=' * 20}")

    dataset = get_gsm8k_questions(num_samples=num_train_samples)

    trainer.train_dataset = dataset

    # Evaluate BEFORE training
    # pre_accuracy, _ = evaluate_model(model, tokenizer, validation_dataset)
    # print(f"Accuracy BEFORE training: {pre_accuracy:.2%}")

    # Train
    trainer.train()

    # Evaluate AFTER training
    post_accuracy, _ = evaluate_model(model, tokenizer, validation_dataset)
    print(f"Accuracy AFTER training: {post_accuracy:.2%}")


# %%

# %%
# Evaluate on random 10-sample set
pre_accuracy_sample, pre_examples = evaluate_model(
    model, tokenizer, validation_dataset, sample_size=1
)
print("\nPerformance BEFORE training (10 random samples):")
for ex in pre_examples:
    print(f"\nQuestion: {ex['question']}")
    print(f"True Answer: {ex['true']}")
    print(f"Predicted Answer: {ex['predicted']}")
    print(f"Correct: {'✅' if ex['correct'] else '❌'}")

print(f"\nAccuracy (random 10 samples) BEFORE training: {pre_accuracy_sample:.2%}")

# %%
# Evaluate on entire validation set
pre_accuracy_full, _ = evaluate_model(model, tokenizer, validation_dataset)
print(f"\nAccuracy (full validation set) BEFORE training: {pre_accuracy_full:.2%}")

# %%
trainer.train()

# %%
# Evaluate on random 10-sample set after training
post_accuracy_sample, post_examples = evaluate_model(
    model, tokenizer, validation_dataset, sample_size=10
)
print("\nPerformance AFTER training (10 random samples):")
for ex in post_examples:
    print(f"\nQuestion: {ex['question']}")
    print(f"True Answer: {ex['true']}")
    print(f"Predicted Answer: {ex['predicted']}")
    print(f"Correct: {'✅' if ex['correct'] else '❌'}")

print(f"\nAccuracy (random 10 samples) AFTER training: {post_accuracy_sample:.2%}")

# Evaluate on entire validation set after training
post_accuracy_full, _ = evaluate_model(model, tokenizer, validation_dataset)
print(f"\nAccuracy (full validation set) AFTER training: {post_accuracy_full:.2%}")
