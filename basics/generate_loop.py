from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "models/qwen3-0.6b"

model = AutoModelForCausalLM.from_pretrained(model_path).to("mps")
tokenizer = AutoTokenizer.from_pretrained(model_path)


def apply_chat_template(prompts: list[str]):
    return [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def get_next_token(logits: torch.Tensor):
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def generate_loop(prompt: str, max_new_tokens: int):
    texts = apply_chat_template([prompt])

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        "mps"
    )
    generated = input_ids["input_ids"]
    attention_mask = input_ids["attention_mask"]
    with torch.inference_mode():
        outputs = model(
            input_ids=generated,
            attention_mask=attention_mask,
            use_cache=True,
        )

    past_key_values = outputs.past_key_values
    for _ in range(max_new_tokens):
        next_token_id = get_next_token(outputs.logits)
        generated = torch.cat([generated, next_token_id], dim=1)
        token_next = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        print(token_next, end="", flush=True)
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        new_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)
        with torch.no_grad():
            outputs = model(
                input_ids=next_token_id,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values

    print()
    return generated


if __name__ == "__main__":
    prompt = "Hello, who are you?"
    generate_loop(prompt, max_new_tokens=300)
