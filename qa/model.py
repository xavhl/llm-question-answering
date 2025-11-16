from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
import torch

def load_generator(model_name, max_new_tokens=150, temperature=0.7, quantize=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # small safety: choose dtype based on availability
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # For quantization: user can enable via env; we leave hook here (bitsandbytes requires extra packages)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map='auto' if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    return generator
