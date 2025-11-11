import re
import torch
from math_verify import parse, verify


def extract_answer(text):
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
	if match:
		return match.group(1).strip()
	return None


def format_reward(completion, **kwargs):
	# 格式 <think>\n...\n</think>\n<answer>\n...\n</answer>
	pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
	if re.match(pattern, completion, re.DOTALL | re.MULTILINE):
		return 1.0
	else:
		return 0.0
	

def tag_count_reward(completion, **kwargs):
    score = 0.0
    if re.search(r'<think>\s*', completion):
        score += 0.25
    if re.search(r'\s*</think>', completion):
        score += 0.25
    if re.search(r'<answer>\s*', completion):
        score += 0.25
    if re.search(r'\s*</answer>', completion):
        score += 0.25
    return score

import re
from math import isclose

def extract_last_number(text):
    # 提取文本中出现的最后一个数字
    text = text.replace('$', '').replace('%', '')
    pattern = r'(-?\d*\.?\d+(?:e[-+]?\d+)?)'
    numbers = re.findall(pattern, text, flags=re.IGNORECASE)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            return None
    return None

def extract_single_number(text):
    # 只在文本中包含唯一一个数字时，返回它的 float 值，否则返回 None
    pattern = r'(-?\d*\.?\d+(?:e[-+]?\d+)?)'
    numbers = re.findall(pattern, text, flags=re.IGNORECASE)
    if len(numbers) == 1:
        try:
            return float(numbers[0])
        except:
            return None
    return None

def accuracy_reward(completion, solution, **kwargs):
    # 提取答案
    full_answer_content = extract_answer(completion)
    if full_answer_content is None:
        return 0.0

    # 1. 完全匹配
    if str(full_answer_content).strip() == str(solution).strip():
        return 1.0  # exact match reward

    # 2. 单数字匹配
    pred_num = extract_single_number(str(full_answer_content))
    sol_num = extract_single_number(str(solution))
    if pred_num is not None and sol_num is not None and isclose(pred_num, sol_num, rel_tol=1e-6):
        return 1.0  # numeric match reward

    # 3. 最后数字匹配
    pred_num = extract_last_number(str(full_answer_content))
    sol_num = extract_last_number(str(solution))
    if pred_num is not None and sol_num is not None and isclose(pred_num, sol_num, rel_tol=1e-6):
        return 1.0  # last number match reward

    # 4. 不匹配
    return 0.0

def compute_grpo_reward(completions, solutions, reward_funcs, reward_weights=None):
	if reward_weights is None:
		reward_weights = [1.0/len(reward_funcs)] * len(reward_funcs)

	assert len(reward_weights) == len(reward_funcs), "reward_weight and reward_funcs must have the same length"

	rewards_per_sample_per_func = torch.zeros(len(completions), len(reward_funcs))

	for i, (a_completion, a_solution) in enumerate(zip(completions, solutions)):
		for j, reward_func in enumerate(reward_funcs):
			rewards_per_sample_per_func[i, j] = reward_func(a_completion, solution=a_solution)

	reward_weight_tensor = torch.tensor(reward_weights)
	reward_per_completion = (rewards_per_sample_per_func * reward_weight_tensor).sum(dim=1)

	# return avergaed score of different reward functions
	reward_per_reward_func = rewards_per_sample_per_func.mean(dim=0)

	return reward_per_completion, reward_per_reward_func


def compute_group_advantage(reward_per_sample: torch.Tensor, num_generations: int=None, eps: float = 1e-8, scale_rewards: bool = True):
	# 在group内，计算每个生成样本的advantage
	if num_generations is None:
		num_generations = reward_per_sample.shape[0]

	mean_grouped_rewards = reward_per_sample.view(-1, num_generations).mean(dim=1) # 每组平均reward
	std_grouped_rewards = reward_per_sample.view(-1, num_generations).std(dim=1) # 每组的 reward 标准差
	
	# 将 mean 和 std 重复 num_generations 次，以便与 rewards 的形状匹配
	mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
	std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations, dim=0)
    
    # Advantage = 当前样本的 reward − 该组平均 reward
	group_advantage = reward_per_sample - mean_grouped_rewards
	if scale_rewards:
		group_advantage /= (std_grouped_rewards + eps)

	return group_advantage



if __name__ == "__main__":
	# 用于测试
	completions = [
		"<think>\nLet's solve this step by step. Jamie's last name is 'Grey', which has 4 letters. If Bobbie's last name were to be halved, it would be twice the length of Grey, meaning it would be 8 letters long. Therefore, Bobbie’s last name has 8 letters. If Samantha's last name has 3 fewer letters than Bobbie's, we’d subtract 3 from Bobbie's last name length. Hence, Samantha's last name has 5 letters.\n</think>\n<answer>\n5\n</answer>",
		'<think>\nTo solve this problem, let\'s start by identifying the number of letters in each last name:\n\n1. Jamie\'s last name is "Grey," which has 4 letters.\n2. If Bobbie takes two letters off her last name, her last name would be half the length of Jamie\'s name. Since Jamie\'s name has 4 letters, Bobbie\'s new last name would have 4 / 2 = 2 letters.\n3. Bobbie’s last name has 2 letters less than Samantha’s last name. So, Samantha’s last name would have 2 + 2 = 4 letters.\n</think>\n<answer>\nSamantha’s last name has 7 letters.\n</answer>',
		'To solve this problem, let\'s start by identifying the number of letters in each last name:\n\n1. Jamie\'s last name is "Grey," which has 4 letters.\n2. If Bobbie takes two letters off her last name, her last name would be half the length of Jamie\'s name. Since Jamie\'s name has 4 letters, Bobbie\'s new last name would have 4 / 2 = 2 letters.\n3. Bobbie’s last name has 2 letters less than Samantha’s last name. So, Samantha’s last name would have 2 + 2 = 4 letters.\n<answer>\nSamantha’s last name has 7 letters.\n</answer>',

	]
	solutions = ['7', '7']
	reward_funcs = [format_reward, tag_count_reward, accuracy_reward]

	reward_per_sample, reward_per_reward_func = compute_grpo_reward(completions, solutions, reward_funcs)
	print(reward_per_sample)
	print(reward_per_reward_func)
	