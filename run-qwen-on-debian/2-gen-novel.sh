# 检查是否在 git 根目录运行
if [ ! -d .git ] || [ "$(git rev-parse --show-toplevel)" != "$(pwd -P)" ]; then
  echo "错误：请在 git 仓库的根目录下运行此脚本。"
  echo "例如：cd /path/to/your/git/repo && sh run-qwen-on-debian/2-gen-novel.sh"
  exit 1
fi

. ./qwen_env/bin/activate
python run-qwen-on-debian/gen-novel.py