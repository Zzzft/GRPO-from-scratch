import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from prepare_data import maybe_apply_chat_template

def get_per_token_log_probs(
	model: AutoModelForCausalLM,
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
):
	target_ids = input_ids[:, 1:] # label shift：给定前面的 token，预测下一个 token
	logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
	log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
	per_token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
	return per_token_log_probs


def create_completion_mask(completion_ids, eos_token_id):
	is_eos = completion_ids == eos_token_id
	eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
	mask_exists = is_eos.any(dim=1)
	eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
	sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
	return (sequence_indices <= eos_idx.unsqueeze(1)).int()

@torch.no_grad()
def generate_rollouts(
	model: AutoModelForCausalLM, 
	tokenizer: AutoTokenizer, 
	prompts: list[list[dict[str, str]]] | list[str], # prompts maybe a list of list or list of str
	num_of_roullout:int=8,
	max_length: int = 1024,
	temperature: float = 1.0,
	top_p: float = 1.0,
	top_k: int = 50,
	):
	
	model.eval()
	device = model.device

	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id

	prompts = [
		maybe_apply_chat_template(a_prompt, tokenizer)
		for a_prompt in prompts
	]
    
	model_inputs = tokenizer(
		prompts,
		return_tensors="pt",
		padding=True,
		padding_side="left",
		return_attention_mask=True,
	).to(device)

	model_inputs["input_ids"] = model_inputs["input_ids"].repeat_interleave(num_of_roullout, dim=0)
	model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat_interleave(num_of_roullout, dim=0)
	prompt_length = model_inputs["input_ids"].shape[1] 
	
	generation_config = GenerationConfig(
		do_sample=True,
		top_p=top_p,
		top_k=top_k,
		temperature=temperature,
		max_length=max_length,
		pad_token_id=tokenizer.pad_token_id,
	)
    
	sequence_ids = model.generate(
		**model_inputs, 
		generation_config=generation_config
	)

	completions = tokenizer.batch_decode(
		sequence_ids[:, prompt_length:], skip_special_tokens=True
	)

	completion_mask = torch.zeros_like(sequence_ids, dtype=torch.int64)
	partial_completion_mask = create_completion_mask(sequence_ids[:, prompt_length:], tokenizer.eos_token_id)
	completion_mask[:, prompt_length:] = partial_completion_mask

	sequence_mask = torch.cat([model_inputs["attention_mask"], partial_completion_mask], dim=1)

	return sequence_ids, sequence_mask, completion_mask, completions



def get_grpo_loss(
	model: AutoModelForCausalLM,
	sequence_ids: torch.Tensor,
	sequence_mask: torch.Tensor,
	completion_mask: torch.Tensor,
	advantage_per_sample: torch.Tensor,
	prob_per_token_old: torch.Tensor,
	prob_per_token_reference: torch.Tensor,
	epsilon: float,
	beta: float = 0.04,
):
	
	prob_per_token_policy = get_per_token_log_probs(
		model,
		input_ids=sequence_ids,
		attention_mask=sequence_mask,
	)

	coef_1 = (prob_per_token_policy - prob_per_token_old).exp()
	coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
	loss_per_token_1 = coef_1 * advantage_per_sample.unsqueeze(1)
	loss_per_token_2 = coef_2 * advantage_per_sample.unsqueeze(1)
	loss_per_token = -torch.min(loss_per_token_1, loss_per_token_2)

	kl_divergence_per_token = (prob_per_token_policy - prob_per_token_reference).exp() - (prob_per_token_policy - prob_per_token_reference) - 1
	loss_per_token += beta * kl_divergence_per_token

	loss_per_completion = (loss_per_token * completion_mask[:, 1:]).sum(dim=1)
	length_per_completion = completion_mask[:, 1:].sum(dim=1).clamp(min=1)
	loss = (loss_per_completion / length_per_completion).mean()

	return loss


