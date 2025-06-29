from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# -------------------------- 1. 模型和tokenizer加载 --------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
model_name_or_path = os.path.join(script_dir, "qwen-7b")

# -- 检查本地模型路径是否存在 --
if not os.path.isdir(model_name_or_path):
    raise FileNotFoundError(f"本地模型文件夹不存在: {model_name_or_path}")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
)
model.eval()

# -------------------------- 2. 小说大纲和设定 --------------------------
novel_outline = [
    {
        "chapter": 1,
        "title": "异世惊梦",
        "prompt": "你是一位优秀的网络小说家，请根据以下设定和要求，创作小说《穿越之我是技术总监》的第一章。\n\n[故事背景]\n主角：林小夏，现代某互联网公司的技术总监，独立、聪慧、理性。\n核心设定：她在一起服务器机房的意外触电事故中，穿越到了一个类似中国古代的王朝，成为了皇帝身边一位不受宠的妃子。\n\n[本章要求]\n- 核心情节：详细描写林小夏从昏迷中醒来，通过观察周围环境（古色古香的房间、自己的华丽但陈旧的服饰、模糊的铜镜倒影），逐步意识到自己穿越了的震惊和迷茫过程。\n- 关键冲突：皇帝突然驾到，言语轻佻，林小夏用现代人的平等视角和对方进行了简短但充满火药味的对话，为后续关系埋下伏笔。\n- 字数要求：800字左右。\n\n请开始创作第一章："
    },
    {
        "chapter": 2,
        "title": "初显锋芒",
        "prompt": "你正在续写小说《穿越之我是技术总监》。\n\n[故事背景回顾]\n主角林小夏，一位现代技术总监，已穿越成古代妃子，并与皇帝初次交锋，展现了她的与众不同。\n\n[已完成的第一章内容]\n{previous_chapter_content}\n\n[本章要求]\n- 核心情节：宫中爆发小规模的“时疫”（类似流感），人心惶惶。林小夏利用自己的项目管理知识和卫生常识（如隔离、通风、石灰消毒等），迅速制定了一套行之有效的防疫流程，并说服了持怀疑态度的太医和宫女们执行。\n- 关键进展：防疫措施效果显著，疫情很快得到控制。皇帝开始对这位“奇怪”的妃子产生了一丝好奇和赞赏。\n- 字数要求：800字左右。\n\n请开始创作第二章："
    }
]

# -------------------------- 3. 生成函数（重构版） --------------------------
def generate_chapter(prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings - max_new_tokens).to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.75,
        top_p=0.8,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 只解码生成的部分，不包含prompt
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

# -------------------------- 4. 主程序：按大纲生成小说 --------------------------
def main():
    full_novel_content = ""
    previous_chapter_text = ""
    print("--- 开始生成小说《穿越之我是技术总监》 ---\n")

    for chapter_info in novel_outline:
        print(f"--- 正在生成第 {chapter_info['chapter']} 章：{chapter_info['title']} ---")
        
        # 构建当前章节的prompt
        current_prompt = chapter_info["prompt"].replace("{previous_chapter_content}", previous_chapter_text)
        
        # 生成章节内容
        chapter_content = generate_chapter(current_prompt)
        
        # 打印并保存
        print(chapter_content)
        print("\n" + "-"*20 + "\n")
        
        full_novel_content += f"## 第 {chapter_info['chapter']} 章：{chapter_info['title']}\n\n{chapter_content}\n\n"
        previous_chapter_text = chapter_content # 为下一章做准备

    # 可以选择将完整小说保存到文件
    with open("穿越之我是技术总监.txt", "w", encoding="utf-8") as f:
        f.write(full_novel_content)
    
    print("--- 小说生成完毕，已保存至 `穿越之我是技术总监.txt` ---")

if __name__ == "__main__":
    main()
