from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------------- 1. 模型和tokenizer加载（强制设置pad_token） --------------------------
model_path = "./qwen-7b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 关键：直接指定pad_token为Qwen的默认eos_token字符串（<|endoftext|>）
tokenizer.pad_token = "<|endoftext|>"
# 手动确保pad_token_id被正确赋值（避免tokenizer未自动转换）
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.eval()

# -------------------------- 2. 生成函数 --------------------------
def generate_novel(prompt, max_new_tokens=1000, num_turns=3):
    full_text = prompt
    for _ in range(num_turns):
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,  # 此时pad_token已明确设置，可安全填充
            truncation=True,
            max_length=model.config.max_position_embeddings - max_new_tokens,
            return_attention_mask=True
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.95,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,  # 显式传入已设置的pad_token_id
            eos_token_id=tokenizer.eos_token_id
        )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return full_text

# -------------------------- 3. 测试生成 --------------------------
prompt = "林小夏在车祸中醒来，发现自己躺在古代的木床上……"
novel = generate_novel(prompt.strip(), max_new_tokens=1000, num_turns=3)
print(novel)
