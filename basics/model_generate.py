import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/qwen3-0.6b"

model = AutoModelForCausalLM.from_pretrained(model_path).to("mps")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# print(tokenizer.vocab_size) 151643
# print(tokenizer.eos_token_id) 151645
# print(tokenizer.bos_token_id) None
# print(tokenizer.pad_token_id) 151643

prompts = [
    "A City in German with most diversities and cultures and history, what is that city? Answer in format: <answer>city_name</answer>",
    "A Country that are in the European Union and start the World War 2, what is that country's capital? Answer in foramt: <answer>city_name</answer>",
]


def apply_chat_template(prompts: List[str]):
    return [
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


texts = apply_chat_template(prompts)

tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

input_ids = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
    "mps"
)
with torch.no_grad():
    output_ids = model.generate(
        **input_ids,
        max_new_tokens=20,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

print(output_ids.shape)  # [batch_size, seq_len]
input_len = input_ids["input_ids"].shape[1]
print(input_ids)
print(input_ids["input_ids"])
print(input_ids["input_ids"].shape)
print(input_ids["attention_mask"].sum(dim=1))
print(tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True))

"""
Input: List[prompt] -> tokenizer.apply_chat_template(prompts) -> <|im_start|>user\nHello, who are you?<|im_end|> ...
tokenize with padding_side="left": [pad_token_id, pad_token_id, bos_token_id, ...], batch padded to same length
input_len = input_ids["input_ids"].shape[1]

tokenize -> {"input_ids", "attention_mask"}

model.generate -> [batch_size, seq_len], bunch of token_ids

tokenizer -> decode -> to corresponding text
"""
