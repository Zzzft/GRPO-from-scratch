import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import re
from math import isclose

BASE_MODEL = "/inspire/hdd/project/elderlyemotions/kangchun-240108120076/fz/models/Qwen2.5-1.5B-Instruct"
GRPO_MODEL = "/inspire/hdd/project/elderlyemotions/kangchun-240108120076/fz/GRPO/src/GRPO"
DATASET_PATH = "/inspire/hdd/project/elderlyemotions/kangchun-240108120076/fz/hands-on-grpo-from-scratch/grade-school-math/grade_school_math/data/test.jsonl"
SAVE_DIR     = "/inspire/hdd/project/elderlyemotions/kangchun-240108120076/fz/GRPO/eval_results"
MAX_SAMPLES  = 240
TEMPERATURE  = 0.7
TOP_P        = 0.9
MAX_NEW_TOKENS = 512

# ---- system prompt ----
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.,
<think>
 reasoning process here 
</think>
<answer>
 answer here 
</answer>.
"""

def extract_answer(text):
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def format_reward(completion, **kwargs):
    has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL | re.IGNORECASE))
    has_answer = bool(re.search(r'<answer>.*?</answer>', completion, re.DOTALL | re.IGNORECASE))
    return 1.0 if (has_think and has_answer) else 0.0

def tag_count_reward(completion, **kwargs):
    score = 0.0
    text = completion.lower()
    if '<think>' in text: score += 0.25
    if '</think>' in text: score += 0.25
    if '<answer>' in text: score += 0.25
    if '</answer>' in text: score += 0.25
    return score

def extract_last_number(text):
    text = text.replace('$', '').replace('%', '')
    matches = re.findall(r'-?\d*\.?\d+', text)
    return float(matches[-1]) if matches else None


def extract_single_number(text):
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[0]) if len(numbers) == 1 else None

def accuracy_reward(completion, solution, **kwargs):
    predicted = extract_answer(completion)
    if predicted is None:
        return 0.0

    try:
        pred_num = extract_single_number(str(predicted))
        sol_num = extract_single_number(str(solution))
        if pred_num is not None and sol_num is not None and isclose(pred_num, sol_num, rel_tol=1e-6):
            return 1.0

        pred_num = extract_last_number(str(predicted))
        sol_num = extract_last_number(str(solution))
        if pred_num is not None and sol_num is not None and isclose(pred_num, sol_num, rel_tol=1e-6):
            return 1.0

        if str(predicted).strip() == str(solution).strip():
            return 1.0
    except Exception:
        pass
    return 0.0

def compute_grpo_reward(completions, solutions, reward_funcs, reward_weights=None):
    if reward_weights is None:
        reward_weights = [1.0/len(reward_funcs)] * len(reward_funcs)
    rewards = torch.zeros(len(completions), len(reward_funcs))
    for i, (c, s) in enumerate(zip(completions, solutions)):
        for j, f in enumerate(reward_funcs):
            rewards[i, j] = f(c, solution=s)
    reward_weights = torch.tensor(reward_weights)
    reward_per_completion = (rewards * reward_weights).sum(dim=1)
    reward_per_func = rewards.mean(dim=0)
    return reward_per_completion, reward_per_func

def load_dataset(path, max_samples=MAX_SAMPLES):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data[:max_samples]

def generate_responses(model, tokenizer, questions, device):
    responses = []
    model.eval()

    for q in tqdm(questions, desc="Generating"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = outputs[0, inputs.shape[-1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        text = re.sub(r"^.*?(<think>)", r"\1", text, flags=re.DOTALL)
        responses.append(text)
    return responses

def evaluate_model(model_path, dataset, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n===== Evaluating {model_name} =====")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    questions = [item["question"] for item in dataset]
    answers = [item["answer"] for item in dataset]

    responses = generate_responses(model, tokenizer, questions, device)

    reward_funcs = [format_reward, tag_count_reward, accuracy_reward]
    reward_weights = [0.5, 0.5, 1.0]
    reward_per_completion, reward_per_func = compute_grpo_reward(responses, answers, reward_funcs, reward_weights)

    correct = 0
    total = len(responses)
    for i, (q, gt, resp) in enumerate(zip(questions, answers, responses)):
        pred = extract_answer(resp)
        pred_num = extract_last_number(str(pred))
        gt_num = extract_last_number(str(gt))
        is_correct = (pred_num is not None and gt_num is not None and isclose(pred_num, gt_num, rel_tol=1e-6))
        if is_correct:
            correct += 1
        print(f"\n[{i+1}] Question: {q}")
        print(f"Expected: {gt_num}")
        print(f"Predicted: {pred_num}")
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print("-"*100)

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    print("="*100)

    mean_reward = reward_per_completion.mean().item()
    metrics = {
        "mean_reward": mean_reward,
        "format_reward": reward_per_func[0].item(),
        "tag_count_reward": reward_per_func[1].item(),
        "accuracy_reward": reward_per_func[2].item(),
        "true_accuracy": accuracy,
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{model_name}_eval.json")
    results = [{"question": q, "ground_truth": gt, "response": r} for q, gt, r in zip(questions, answers, responses)]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {save_path}")
    print(f"Metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH)
    
    evaluate_model(BASE_MODEL, dataset, "base")
    evaluate_model(GRPO_MODEL, dataset, "grpo")
    #evaluate_model(PPO_MODEL,  dataset, "ppo")