# 更新系统并安装依赖
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv git

# 创建虚拟环境
python3 -m venv ~/qwen_env
source ~/qwen_env/bin/activate

# 安装 CPU 优化的 PyTorch（启用 MKL 加速）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 Hugging Face 库
pip install tiktoken transformers accelerate sentencepiece  # Qwen 依赖 sentencepiece
pip install einops transformers_stream_generator
pip install huggingface-hub

# 克隆 Qwen-7B 模型（需科学上网或镜像）
huggingface-cli download Qwen/Qwen-7B --local-dir ./qwen-7b --resume-download