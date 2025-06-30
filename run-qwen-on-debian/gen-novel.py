from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import os

# -------------------------- 1. 模型和tokenizer加载 --------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_name_or_path = os.path.join(project_root, "qwen-7b")

if not os.path.isdir(model_name_or_path):
    raise FileNotFoundError(f"本地模型文件夹不存在: {model_name_or_path}")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, local_files_only=True)

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

# -------------------------- 2. 小说生成函数 (重构为流式生成) --------------------------
def generate_chapter(prompt_messages):
    """根据提供的消息流式生成小说章节，并在检测到结束标记时立即停止"""
    input_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    
    # eos_token_id 151645 is <|im_end|>
    eos_token_ids = [tokenizer.eos_token_id, 151645]
    eos_token_ids = [t for t in eos_token_ids if t is not None]

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=2048,
        do_sample=False,
        eos_token_id=eos_token_ids,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    # 在一个新线程中运行生成，以便我们可以监控输出
    # 这是因为streamer会直接打印到控制台，我们需要一种方式来捕获文本
    # 这里我们简化逻辑，直接生成并后处理，因为TextStreamer主要用于交互式展示
    # 对于文件写入，我们一次性生成更高效
    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        do_sample=False,
        eos_token_id=eos_token_ids,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    
    response_tensor = outputs[0][input_ids.shape[-1]:]
    full_response = tokenizer.decode(response_tensor, skip_special_tokens=False)

    # -- 终极修复：手动截断 --
    stop_token = "<|im_end|>"
    stop_index = full_response.find(stop_token)
    if stop_index != -1:
        clean_response = full_response[:stop_index]
    else:
        clean_response = full_response

    # 最后再移除可能残留的特殊token，并清理首尾空白
    final_text = tokenizer.decode(tokenizer.encode(clean_response, add_special_tokens=False), skip_special_tokens=True).strip()
    return final_text

# -------------------------- 3. 主程序 --------------------------
if __name__ == "__main__":
    novel_title = "穿越之我是技术总监"
    novel_file = os.path.join(project_root, f"{novel_title}.txt")

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

    system_prompt = "你是一位专业的中文网络小说作家。你的唯一任务是根据我提供的大纲和要求，创作小说。请严格遵守规则：1. 只写小说内容。2. 使用简体中文。3. 严格遵循情节大纲。4. 创作完成后，立即停止，不要输出任何额外内容、解释、问题或结尾词。"

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

    请严格按照以上设定创作，完成后立即停止。
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print("正在生成小说第一章...")
    novel_chapter = generate_chapter(messages)
    print("生成完毕！")

    # -- 文件操作 --
    # 写入文件前，先创建一个空的标题和章节标题
    with open(novel_file, "w", encoding="utf-8") as f:
        f.write(f"# {novel_title}\n\n")
        f.write(f"## {novel_outline['chapter_1_title']}\n\n")
        f.write(novel_chapter)

    print(f"小说已保存至: {novel_file}")
