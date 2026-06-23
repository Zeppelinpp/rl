import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/qwen3-0.6b"

model = AutoModelForCausalLM.from_pretrained(model_path).to("mps")
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello, who are you?"}],
    tokenize=False,
    add_generation_prompt=True,
)

input_ids = tokenizer(prompt, return_tensors="pt").to("mps")
with torch.no_grad():
    next_token_logits = model(**input_ids, use_cache=True, output_hidden_states=True)

# model forward output -> CausalLMOutputWithPast(loss, logits, past_key_values, hidden_states, attentions)
print(next_token_logits.logits.shape)  # [batch_size, seq_len, vocab_size]
# input [t1, t2, t3, ..., tn], t2 is predictions of t2 based on t1, ...
# when doing inference we only need the last prediction as the new token prediction
print(next_token_logits.logits[:, -1, :].shape)  # [batch_size, vocab_size]
print(
    f"hidden_states len: {len(next_token_logits.hidden_states)}"
)  # embedding ouputs + each layer output
print(
    torch.allclose(
        model.lm_head(next_token_logits.hidden_states[-1]),
        next_token_logits.logits,
        atol=1e-3,
        rtol=1e-3,
    )
)
# print(next_token_logits.past_key_values) -> DynamicCache(layers=[num layers of DynamicLayer])

new_token_id = torch.argmax(next_token_logits.logits[:, -1, :], dim=-1, keepdim=True)
new_token = tokenizer.decode(new_token_id[0])
print(f"new_token: {new_token}")

with torch.no_grad():
    new_next_token_logits = model(
        input_ids=new_token_id,
        past_key_values=next_token_logits.past_key_values,
        use_cache=True,
    )
new_new_token_id = torch.argmax(
    new_next_token_logits.logits[:, -1, :], dim=-1, keepdim=True
)
new_new_token = tokenizer.decode(new_new_token_id[0])
print(f"new_new_token: {new_new_token}")

"""
Model __call__ accepts:
outputs = model(
    input_ids=...,
    attention_mask=...,
    past_key_values=...,
    use_cache=True,
    output_hidden_states=True,
    output_attentions=False,
)

model forward output -> CausalLMOutputWithPast(loss, logits, past_key_values, hidden_states, attentions)
outputs.logits -> [batch_size, seq_len, vocab_size], last one is the new token prediction
outputs.hidden_states -> [batch_size, seq_len, hidden_size], embedding ouputs + each layer output
lm_head(outputs.hidden_states[-1]) -> logits of the last layer
decoding phase can use kv cache to speed up inference

argmax: dim=-1 is do argmax along the last dimension which is the vocab_size (each token's probability)
- with no keepdim, the result is a tensor of shape [batch_size], keepdim=True makes it a tensor of shape [batch_size, 1]
- model forward needs a [batch_size, seq_len] tensor
"""
