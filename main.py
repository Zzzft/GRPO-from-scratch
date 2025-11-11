import json, re, torch, copy, wandb
from tqdm import tqdm
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_cosine_schedule_with_warmup

from prepare_data import prepare_dataloader
from reward import format_reward, tag_count_reward, accuracy_reward, compute_grpo_reward, compute_group_advantage
from grpo_utils import generate_rollouts, get_per_token_log_probs, get_grpo_loss

def train_with_grpo(
    model_policy, # 策略模型
    tokenizer, 
    train_dataloader, 
    eval_dataloader,
    reward_funcs,
    reward_weights,
    n_epoch=1, # 训练轮数
    n_roullout=8, # 每组rollout数量
    max_length=1024, # 最大生成长度
    batch_size_micro=2, # 微批处理大小
    batch_size_micro_for_no_grad=4, # 无梯度计算的微批处理大小
    learning_rate=5e-6, # 学习率
    epsilon=0.2, # PPO裁剪阈值
    beta=0.001, # KL散度系数
    mu=2, # 每个批次的策略更新次数
    device=None, 
    model_name_or_path=None,
    project_name="GRPO"
):  
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    optimizer = torch.optim.AdamW(model_policy.parameters(), lr=learning_rate)

    total_steps = len(train_dataloader) * n_epoch
    warmup_steps = int(total_steps * 0.1)  # 10% warm up
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    step = 0
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(n_epoch):
        print(f"Epoch {epoch + 1}/{n_epoch}")
        model_reference = copy.deepcopy(model_policy)
        model_reference.eval()
        for param in model_reference.parameters():
            param.requires_grad = False

        for batch in tqdm(train_dataloader, desc="Training", total=len(train_dataloader), leave=True):
            prompts = [example['prompt'] for example in batch]
            solutions = [example['solution'] for example in batch]

            sequence_ids, sequence_mask, completion_mask, completions = generate_rollouts(
                model_policy, 
                tokenizer, 
                prompts, 
                num_of_roullout=n_roullout, 
                max_length=max_length, 
                temperature=0.7,
                top_p=0.9, 
                top_k=50,
            )

            solutions = [s for s in solutions for _ in range(n_roullout)]

            reward_per_completion, reward_per_reward_func = compute_grpo_reward(
                completions, 
                solutions, 
                reward_funcs,
                reward_weights,
            )

            group_advantage_per_sample = compute_group_advantage(
                reward_per_completion
            ).to(device)

            # 计算log prob
            with torch.no_grad():
                prob_per_token_old = []
                prob_per_token_reference = []
                
                for i in range(0, len(sequence_ids), batch_size_micro_for_no_grad):
                    sequence_ids_batch = sequence_ids[i:i + batch_size_micro_for_no_grad]
                    sequence_mask_batch = sequence_mask[i:i + batch_size_micro_for_no_grad]
                    
                    prob_old_batch = get_per_token_log_probs(
                        model_policy,  # 使用当前policy作为old policy
                        input_ids=sequence_ids_batch,
                        attention_mask=sequence_mask_batch,
                    )
                    prob_ref_batch = get_per_token_log_probs(
                        model_reference,
                        input_ids=sequence_ids_batch,
                        attention_mask=sequence_mask_batch,
                    )
                    
                    prob_per_token_old.append(prob_old_batch)
                    prob_per_token_reference.append(prob_ref_batch)
                
                # mini batch: 优化显存
                prob_per_token_old = torch.cat(prob_per_token_old, dim=0)
                prob_per_token_reference = torch.cat(prob_per_token_reference, dim=0)

            loss_list = []
            
            for update_step in range(mu):
                optimizer.zero_grad()
                tmp = [] 
                
                for i in range(0, len(sequence_ids), batch_size_micro):
                    sequence_ids_batch = sequence_ids[i:i + batch_size_micro]
                    sequence_mask_batch = sequence_mask[i:i + batch_size_micro]
                    completion_mask_batch = completion_mask[i:i + batch_size_micro]
                    group_advantage_per_sample_batch = group_advantage_per_sample[i:i + batch_size_micro]

                    prob_per_token_old_batch = prob_per_token_old[i:i + batch_size_micro]
                    prob_per_token_reference_batch = prob_per_token_reference[i:i + batch_size_micro]
                        
                    loss = get_grpo_loss(
                        model_policy,
                        sequence_ids_batch,
                        sequence_mask_batch,
                        completion_mask_batch,
                        group_advantage_per_sample_batch,
                        prob_per_token_old_batch,
                        prob_per_token_reference_batch,
                        epsilon,
                        beta
                    )
                    loss.backward()
                    tmp.append(loss.item())
                
                optimizer.step()
                loss_list.append(np.mean(tmp).item())

            rewards = {
                k: v.item() 
                for k, v in zip(
                    [item.__name__ for item in reward_funcs], 
                    reward_per_reward_func
                )
            }
            rewards['mean_reward'] = np.mean(list(rewards.values())).item()
            log_info = {
                'epoch': epoch,
                'step': step,
                'loss': loss_list[-1] if loss_list else 0.0,  
                'learning_rate': scheduler.get_last_lr()[0],
                **rewards
            }
            step += 1

            scheduler.step()  
            tqdm.write(str(log_info))
            wandb.log(log_info)

    if model_name_or_path:
        model_save_name = model_name_or_path.split('/')[-1]
    else:
        model_save_name = "trained_model"
    
    model_save_path = f"./grpo_{model_save_name}_{start_time}_epoch_{n_epoch}"
    model_policy.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    return model_policy, tokenizer

# 主程序
if __name__ == "__main__":
    # 超参数设置
    n_epoch = 1
    n_roullout = 8
    max_length = 1024
    batch_size_dataloader = 4  # 每个步骤的批次数
    batch_size_micro = 2  # 计算GRPO损失并反向传播时的微批大小
    batch_size_micro_for_no_grad = 4  # 计算旧策略/reference模型log-prob时使用的批大小
    learning_rate = 5e-6
    epsilon = 0.2  # PPO比率裁剪阈值
    beta = 0.001  # KL散度系数
    mu = 2  # 每个批次的策略更新次数

    model_name_or_path = "/inspire/hdd/project/elderlyemotions/kangchun-240108120076/fz/models/Qwen2.5-1.5B-Instruct"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using primary device: {device}")

    wandb.init(project="GRPO")

    print("Downloading model...")
    model_policy = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype='auto',
    ).to(device)
    print("Model downloaded")
    model_policy.config.use_cache = False
    model_policy.gradient_checkpointing_enable() 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model_policy.config.pad_token_id = tokenizer.eos_token_id
    model_policy.config.eos_token_id = tokenizer.eos_token_id

    train_dataloader, eval_dataloader = prepare_dataloader(tokenizer, batch_size=batch_size_dataloader)
    reward_funcs = [format_reward, tag_count_reward, accuracy_reward]
    reward_weights = [0.5, 0.5, 1.0]

    # 调用训练函数
    trained_model, trained_tokenizer = train_with_grpo(
        model_policy=model_policy,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        n_epoch=n_epoch,
        n_roullout=n_roullout,
        max_length=max_length,
        batch_size_micro=batch_size_micro,
        batch_size_micro_for_no_grad=batch_size_micro_for_no_grad,
        learning_rate=learning_rate,
        epsilon=epsilon,
        beta=beta,
        mu=mu,
        device=device,
        model_name_or_path=model_name_or_path
    )

    wandb.finish()