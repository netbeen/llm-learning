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

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

# -------------------------- 2. 小说生成函数 --------------------------
def generate_chapter(prompt_messages):
    """根据提供的消息生成小说章节"""
    # 使用聊天模板格式化输入
    input_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # -- 关键修复：同时使用默认EOS和Qwen特定的<|im_end|>作为停止符 --
    eos_token_ids = [tokenizer.eos_token_id, 151645]

    # 生成文本
    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        eos_token_id=eos_token_ids,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# -------------------------- 3. 主程序 --------------------------
if __name__ == "__main__":
    novel_title = "穿越之我是技术总监"
    novel_file = os.path.join(project_root, f"{novel_title}.txt")

    # -- 小说情节大纲和角色设定 --
    novel_outline = {
        "title": novel_title,
        "protagonist": {
            "name": "林天宇",
            "background": "现代世界的顶尖软件架构师，技术总监（CTO）",
            "personality": "冷静、理性、逻辑思维能力强，善于解决复杂问题"
        },
        "setting": "一个类似古代中国的世界，但拥有一些独特的能量体系（例如‘气’或‘灵力’），技术水平落后。",
        "plot_summary": "林天宇意外穿越到这个古代世界，成为一个没落贵族家庭的子弟。他利用自己的现代技术知识，结合这个世界的能量体系，开创了一条独特的‘科技修仙’之路，最终成为影响整个世界格局的关键人物。",
        "chapter_1_title": "第一章：异世惊魂"
    }

    # -- System Prompt: 设定模型角色和任务 --
    system_prompt = "你是一位专业的中文网络小说作家，你的任务是根据我提供的大纲和要求，创作一部高质量的穿越题材小说。请严格遵守以下规则：\n1. 只专注于小说创作，不要进行任何与剧情无关的讨论或解释。\n2. 必须使用简体中文进行创作。\n3. 严格遵循我提供的角色设定和情节大纲。\n4. 不要输出任何英文或代码。"

    # -- User Prompt: 提供具体的创作指令 --
    user_prompt = f"""
    请根据以下设定，创作小说《{novel_outline['title']}》的第一章，章节标题为《{novel_outline['chapter_1_title']}》。

    **主角设定**：
    - 姓名：{novel_outline['protagonist']['name']}
    - 背景：{novel_outline['protagonist']['background']}
    - 性格：{novel_outline['protagonist']['personality']}

    **世界背景**：
    {novel_outline['setting']}

    **故事大纲**：
    {novel_outline['plot_summary']}

    请开始创作第一章：
    """

    # -- 构建符合ChatML格式的对话 --
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print("正在生成小说第一章...")
    chapter_content = generate_chapter(messages)
    print("生成完毕！")

    # -- 保存小说内容 --
    with open(novel_file, "w", encoding="utf-8") as f:
        f.write(f"# {novel_outline['title']}\n\n")
        f.write(f"## {novel_outline['chapter_1_title']}\n\n")
        f.write(chapter_content)

    print(f"小说已保存至: {novel_file}")
