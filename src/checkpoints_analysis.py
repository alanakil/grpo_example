# %%
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import gc
import matplotlib.pyplot as plt

# %%
# ---------------------------
# Utility functions
# ---------------------------


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer


def evaluate_model(model, tokenizer, dataset, sample_size=None):
    model.eval()
    correct = 0
    total = 0
    examples = []

    if sample_size:
        dataset = random.sample(list(dataset), sample_size)

    for item in tqdm(dataset, desc="Evaluating"):
        # Prepare prompt using the chat template
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
        # Extract everything after the assistant marker
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

    accuracy = correct / total if total > 0 else 0
    return accuracy, examples


def get_gsm8k_questions(split="test", num_samples=None):
    SYSTEM_PROMPT = """
    You will be given a Question and you must respond in the following format exactly:
    <reasoning>
    Your detailed chain-of-thought here.
    </reasoning>
    <answer>
    Your final answer here. Provide just the number, no additional text.
    </answer>
    """
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {x['question']}"},
            ],
            "answer": x["answer"].split("####")[1].strip()
            if "####" in x["answer"]
            else x["answer"],
        }
    )
    if num_samples:
        data = data.shuffle(seed=42).select(range(num_samples))
    return data


# %%
# ---------------------------
# Main evaluation script
# ---------------------------

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# Define your model name and checkpoint directory
model_name = "Qwen-0.5B-GRPO-Base"  # same as training
output_dir = f"./output/{model_name}"  # directory where checkpoints are saved
tokenizer_dir = output_dir + "/checkpoint-1868"

# Load tokenizer (assumed to be the same across checkpoints)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
tokenizer.pad_token = tokenizer.eos_token

# Load evaluation dataset
validation_dataset = get_gsm8k_questions("test", 300)

# Discover and sort checkpoint directories (e.g., checkpoint-100, checkpoint-200, etc.)
checkpoint_dirs = [
    os.path.join(output_dir, d)
    for d in os.listdir(output_dir)
    if d.startswith("checkpoint")
]
checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))

accuracy_list = []
# Evaluate each checkpoint
for ckpt in checkpoint_dirs:
    print(f"\nEvaluating checkpoint: {ckpt}")
    # Load model from checkpoint and move it to GPU
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, device_map=None
    ).to("cuda")
    accuracy, _ = evaluate_model(model, tokenizer, validation_dataset, sample_size=300)
    print(f"Accuracy for {ckpt}: {accuracy:.2%}")
    accuracy_list.append(accuracy)
    del model
    torch.cuda.empty_cache()
    gc.collect()

# %%
for idx in range(len(checkpoint_dirs)):
    print(f"Checkpoint: {checkpoint_dirs[idx]}")
    print(f"Accuracy: {accuracy_list[idx]}")
    print("----------------------------------")

# %%
# Plot accuracy over checkpoints
x_ticks = [check[-4:] for check in checkpoint_dirs]
plt.plot(x_ticks, accuracy_list, "-o")
plt.xticks(rotation=45)
plt.xlabel("Checkpoint")
plt.ylabel("Accuracy")
plt.show()

# %%
