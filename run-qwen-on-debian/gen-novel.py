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
        "prompt": "你是一位顶级网络小说家，你的任务是续写《穿越之我是技术总监》这部小说的第一章。\n\n**故事开篇:**\n'李明揉了揉酸涩的眼睛，显示器上的代码像一群不知疲倦的蚂蚁，仍在不知疲倦地爬行。作为公司的技术总监，996早已是家常便饭。他端起桌上早已冰凉的咖啡，猛灌一口，试图驱散深夜的困意。就在这时，一股强烈的眩晕感袭来，他眼前的代码瞬间扭曲、碎裂，化为一片黑暗。'\n\n**你的任务:**\n请接着上面的开篇，继续创作第一章【异世惊梦】。\n\n**本章核心要求:**\n1.  **环境描写:** 详细描写李明醒来后所见的古代房间环境。例如：雕花的木床、古朴的桌椅、摇曳的烛火、身上丝质的陌生衣物。强调现代与古代环境的巨大反差给他带来的视觉和心理冲击。\n2.  **内心活动:** 重点刻画李明从困惑、震惊到试图用逻辑分析现状的心理过程。他可能会怀疑是同事的恶作剧，或者是自己加班过度产生的幻觉。\n3.  **初步行动:** 描写他如何从床上坐起，踉跄地走向一面模糊的铜镜，并在镜中看到了一个完全陌生的、属于古代人的面孔，最终让他认识到自己穿越的残酷现实。\n\n请严格按照以上要求，以流畅的文笔和生动的描写，完成第一章的创作。",
    },
    {
        "chapter": 2,
        "title": "初探虚实",
        "prompt": "你正在续写《穿越之我是技术总监》的第二章。\n\n**第一章回顾:**\n{previous_chapter_content}\n\n**你的任务:**\n请根据第一章的内容，创作第二章【初探虚实】。\n\n**本章核心要求:**\n1.  **稳定情绪:** 描写李明在确认穿越的事实后，如何利用他作为技术总监的强大心理素质和逻辑思维能力，强迫自己冷静下来，并开始分析当前的处境。\n2.  **首次互动:** 一个古装打扮的丫鬟推门而入，这是李明与这个世界的第一次正式接触。请设计一段简短但信息量丰富的对话。李明需要通过巧妙的提问（例如：'我睡了多久？'、'这是何处？'），在不暴露自己身份的情况下，获取关于当前时间、地点、以及自己身份的基本信息。\n3.  **信息整合:** 丫鬟离开后，李明需要根据得到的零碎信息，进行推理和分析，对自己的新身份和所处的世界有一个初步的、模糊的认知。\n\n请围绕以上要点展开创作，保持故事的连贯性和逻辑性。",
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
