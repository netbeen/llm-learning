from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# -------------------------- 1. 模型和tokenizer加载 --------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_name_or_path = os.path.join(project_root, "qwen-7b")

# -- 检查本地模型路径是否存在 --
if not os.path.isdir(model_name_or_path):
    raise FileNotFoundError(f"本地模型文件夹不存在: {model_name_or_path}")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True)

# -- 手动设置聊天模板 --
# 解决本地加载时tokenizer.chat_template可能为空的问题
if tokenizer.chat_template is None:
    qwen_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{'<|im_start|>system\n' + message['content'] + '<|im_end|>\n'}}"
        "{% elif message['role'] == 'user' %}"
        "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"
        "{% elif message['role'] == 'assistant' %}"
        "{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n'}}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{'<|im_start|>assistant\n'}}"
        "{% endif %}"
    )
    tokenizer.chat_template = qwen_template
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
        "prompt": "你是一位顶级的中文网络小说家。请以专业的文笔，接着以下故事开篇，续写《穿越之我是技术总监》第一章【异世惊梦】的内容。\n\n**故事开篇:**\n李明揉了揉酸涩的眼睛，显示器上的代码像一群不知疲倦的蚂蚁，仍在不知疲倦地爬行。作为公司的技术总监，996早已是家常便饭。他端起桌上早已冰凉的咖啡，猛灌一口，试图驱散深夜的困意。就在这时，一股强烈的眩晕感袭来，他眼前的代码瞬间扭曲、碎裂，化为一片黑暗。\n\n**续写要求:**\n1.  **环境描写:** 必须详细描写李明醒来后所见的古代房间环境（如雕花木床、古朴桌椅、摇曳烛火、丝质衣物），以此突出古今环境的巨大反差和心理冲击。\n2.  **内心刻画:** 必须重点刻画主角从困惑、震惊到尝试用逻辑分析现状的内心过程，他可能会怀疑是幻觉或恶作剧。\n3.  **关键行动:** 必须描写他起身走向铜镜，并从镜中看到一张完全陌生的古代人面孔，这个情节是让他认识到穿越现实的关键。\n\n请严格遵循以上三点要求，用纯中文进行创作，不要涉及任何无关的讨论或输出英文。"
    },
    {
        "chapter": 2,
        "title": "初探虚实",
        "prompt": "你是一位顶级的中文网络小说家。这是《穿越之我是技术总监》的第二章【初探虚实】的创作任务。\n\n**紧接上一章内容:**\n{previous_chapter_content}\n\n**本章续写要求:**\n1.  **稳定情绪与分析:** 必须描写李明在确认穿越后，如何运用其作为技术总监的强大心理素质和逻辑思维能力，强迫自己冷静并分析处境。\n2.  **首次互动与信息获取:** 必须设计一个丫鬟推门而入的场景。通过这次互动，李明需要通过巧妙的、不暴露身份的提问（如“我睡了多久？”、“这是何处？”），获取关于当前时间、地点和自身身份的关键信息。\n3.  **信息整合与推理:** 丫鬟离开后，必须描写李明整合获取到的零碎信息，并进行逻辑推理，从而对自己的新身份和所处世界形成一个初步认知。\n\n请严格遵循以上三点要求，用纯中文进行创作，确保故事的连贯性和逻辑性。"
    }
]

# -------------------------- 3. 生成函数（重构版） --------------------------
def generate_chapter(prompt, max_new_tokens=1024):
    # 简化prompt结构，直接使用user角色传递所有指令
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用聊天模板
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=model.config.max_position_embeddings - max_new_tokens).to(model.device)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 只解码生成的部分，不包含prompt，这是最稳妥的方式
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
