import torch
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer


app = FastAPI(
    title="ModelHttpServer",
    description="A FastAPI server for serving LLM model",
)

model_path = "models/qwen3-0.6b"
model = AutoModelForCausalLM.from_pretrained(model_path).to("mps")
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: Request):
    global model, tokenizer
    req = await req.json()
    prompt = req["prompt"]
    args = req["args"]

    texts = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        "mps"
    )
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=1000
            if args.get("max_new_tokens") is None
            else args["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )
    input_len = input_ids["input_ids"].shape[1]
    return tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
