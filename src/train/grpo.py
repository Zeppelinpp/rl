from dataclasses import dataclass
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

device = torch.device("mps")


@dataclass
class RolloutSample:
    prompt: str
    responses: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    reward: float
    advantage: float


def reward_fn(response: str):
    text = response.strip()
    lower = text.lower()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if m is None:
        if "berlin" in lower:
            return 0.2
        return 0.0
    answer = m.group(1).strip().lower().strip(".。")
    if answer != "berlin":
        return -0.2
    return 1.0


def get_logprobs(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logprobs = F.log_softmax(
        outputs.logits[:, :-1, :], dim=-1
    )  # [batch_size, seq_len - 1, vocab_size]
    labels = input_ids[:, 1:]  # [batch_size, seq_len - 1]

    return torch.gather(
        logprobs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)  # [batch_size, seq_len - 1]


def compute_advantages_in_place(samples, group_size):
    # Normalize within each prompt's group (GRPO per-prompt baseline).
    for start in range(0, len(samples), group_size):
        group = samples[start : start + group_size]

        rewards = torch.tensor(
            [s.reward for s in group],
            dtype=torch.float32,
            device=device,
        )

        mean = rewards.mean()
        std = rewards.std(unbiased=False)

        advs = (rewards - mean) / (std + 1e-5)

        for sample, adv in zip(group, advs):
            sample.advantage = adv.item()


def compute_kl(new_logprobs, ref_logprobs):
    """Surrogate KL penalty on sampled tokens."""
    log_ratio = (ref_logprobs - new_logprobs).float()
    return torch.exp(log_ratio) - log_ratio - 1


def collate_rollout_samples(samples, tokenizer, device=device):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [s.input_ids for s in samples],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(device)

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [s.attention_mask for s in samples],
        batch_first=True,
        padding_value=0,
    ).to(device)

    response_mask = torch.nn.utils.rnn.pad_sequence(
        [s.response_mask for s in samples],
        batch_first=True,
        padding_value=0.0,
    ).to(device)

    old_logprobs = torch.nn.utils.rnn.pad_sequence(
        [s.old_logprobs for s in samples],
        batch_first=True,
        padding_value=0.0,
    ).to(device)

    ref_logprobs = torch.nn.utils.rnn.pad_sequence(
        [s.ref_logprobs for s in samples],
        batch_first=True,
        padding_value=0.0,
    ).to(device)

    advantages = torch.tensor(
        [s.advantage for s in samples],
        dtype=torch.float32,
        device=device,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "old_logprobs": old_logprobs,
        "ref_logprobs": ref_logprobs,
        "advantages": advantages,
    }


def grpo_loss(model, batch: dict, clip_eps: float, kl_coef: float):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]
    old_logprobs = batch["old_logprobs"]
    ref_logprobs = batch["ref_logprobs"]
    advantages = batch["advantages"]

    # compute loss
    new_logprobs = get_logprobs(model, input_ids, attention_mask)

    ratio = torch.exp(new_logprobs - old_logprobs)
    advantages = advantages.unsqueeze(-1)  # [B, 1]
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

    policy_loss = -torch.min(unclipped, clipped)
    kl_loss = compute_kl(new_logprobs, ref_logprobs)

    loss_per_token = (policy_loss + kl_coef * kl_loss) * response_mask
    token_count_per_sample = response_mask.sum(dim=-1).clamp(min=1.0)
    loss_per_sample = loss_per_token.sum(dim=-1) / token_count_per_sample
    loss = loss_per_sample.mean()
    denom = response_mask.sum().clamp(min=1.0)

    metrics = {
        "policy_loss": ((policy_loss * response_mask).sum() / denom).detach(),
        "kl": ((kl_loss * response_mask).sum() / denom).detach(),
        "loss": loss.detach(),
        "mean_ratio": ((ratio * response_mask).sum() / denom).detach(),
    }
    return loss, metrics


def print_metrics(metrics):
    for k, v in metrics.items():
        if torch.is_tensor(v):
            v = v.item()
        print(f"{k}: {v:.3f}")


def rollout(
    policy_model,
    ref,
    tokenizer,
    prompts: list[str],
    group_size: int,
    max_new_tokens: int = 1000,
):
    """Every prompt generates group_size responses"""
    samples = []
    texts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    repeated_texts = []
    repeated_prompts = []
    for prompt, text in zip(prompts, texts):
        for _ in range(group_size):
            repeated_texts.append(text)
            repeated_prompts.append(prompt)

    enc = tokenizer(
        repeated_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)  # [batch_size * group_size, seq_len]

    generated = policy_model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=1.2,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    ).to(device)  # [batch_size * group_size, seq_len]
    prompt_width = enc["input_ids"].shape[1]
    new_tokens = generated[:, prompt_width:]  # [B, new_len]
    # Keep everything up to and including the first EOS; mask post-EOS pad tokens.
    is_eos = new_tokens == tokenizer.eos_token_id
    keep = (is_eos.cumsum(dim=1) - is_eos.long()) == 0  # [B, new_len] bool
    new_attention_mask = keep.to(enc["attention_mask"].dtype)
    attention_mask = torch.cat([enc["attention_mask"], new_attention_mask], dim=1)

    # get logprobs
    with torch.no_grad():
        old_logprobs = get_logprobs(policy_model, generated, attention_mask)
        ref_logprobs = get_logprobs(ref, generated, attention_mask)

    for i in range(generated.shape[0]):  # batch_size
        # Only decode kept tokens so reward_fn doesn't see post-EOS tokens.
        response = tokenizer.decode(
            generated[i, prompt_width:][keep[i]], skip_special_tokens=True
        )
        # logprob index (prompt_width-1)+k predicts generated token prompt_width+k,
        # so weight response tokens with the same post-EOS keep mask.
        response_mask = torch.zeros(
            generated[i].shape[0] - 1, dtype=torch.float32, device=generated.device
        )
        response_mask[prompt_width - 1 :] = keep[i].float()

        # print(f"Prompt: {repeated_prompts[i]}\nResponse: {response}\n")
        samples.append(
            RolloutSample(
                prompt=repeated_prompts[i],
                responses=response,
                input_ids=generated[i],
                attention_mask=attention_mask[i],
                response_mask=response_mask,
                old_logprobs=old_logprobs[i].detach(),
                ref_logprobs=ref_logprobs[i].detach(),
                reward=reward_fn(response),
                advantage=0.0,
            )
        )
        print("=" * 80)
        print("prompt:", repeated_prompts[i])
        print("response:", repr(response))

    # advantage = r_i - mean(reward) / (std(reward) + 1e-5)
    compute_advantages_in_place(samples, group_size=group_size)
    return samples


def train(
    prompts: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    max_new_tokens: int,
    group_size: int = 5,
    ppo_epochs: int = 1,
):
    model_path = "models/qwen3-0.6b"
    dtype = torch.bfloat16

    policy_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ref = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    ref.eval()

    # froze ref model's parameters
    for p in ref.parameters():
        p.requires_grad_(False)

    policy_model.train()
    ref.eval()
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr)

    for epoch in range(epochs):
        for index in range(0, len(prompts), batch_size):
            batch_prompts = prompts[index : index + batch_size]
            # rollout K samples (old_logprobs are frozen for the PPO inner-loop)
            policy_model.eval()
            rollout_samples = rollout(
                policy_model,
                ref,
                tokenizer,
                batch_prompts,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
            )

            batch = collate_rollout_samples(rollout_samples, tokenizer, device=device)
            avg_reward = sum(s.reward for s in rollout_samples) / len(rollout_samples)

            # PPO inner-loop: reuse the same rollout for several gradient steps so
            # new_logprobs diverge from the frozen old_logprobs and the clip kicks in.
            policy_model.train()
            for ppo_step in range(ppo_epochs):
                loss, metrics = grpo_loss(
                    policy_model,
                    batch,
                    clip_eps=0.2,
                    kl_coef=0.01,
                )
                metrics["avg_reward"] = avg_reward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()

                print(f"Epoch: {epoch} | ppo_step: {ppo_step} --- Metrics:")
                print_metrics(metrics)


if __name__ == "__main__":
    prompts = [
        "What is the capital city of Germany? Final answer should be surrounded by <answer> and </answer> tags.",
        "Germany is a country in Europe. What is its capital city? Final answer should be surrounded by <answer> and </answer> tags.",
    ]
    train(prompts, epochs=5, batch_size=2, lr=5e-5, max_new_tokens=100)
