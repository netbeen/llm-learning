# Python 虚拟环境 (venv) 核心知识点

## 1. 什么是虚拟环境？

Python 虚拟环境是一个独立的目录，它包含特定版本的 Python 解释器和一套独立的库。其核心思想是为每个项目创建一个隔离的“沙盒”，以避免依赖冲突。

## 2. 为什么需要虚拟环境？

- **避免依赖冲突**：不同项目可能需要同一库的不同版本，虚拟环境可以确保它们互不干扰。
- **保持系统纯净**：避免将所有依赖都安装到全局 Python 环境中，导致其混乱和臃肿。
- **方便协作与部署**：通过 `requirements.txt` 文件，可以轻松地在其他机器上复现完全相同的环境。

## 3. `venv` 核心命令 (macOS / Linux)

`venv` 是 Python 3.3+ 内置的官方虚拟环境管理工具。

### a. 创建环境

在项目根目录下执行，`qwen_env` 是自定义的环境名称。
```bash
python3 -m venv qwen_env
```

### b. 激活环境

激活后，终端提示符前会显示环境名称。
```bash
source qwen_env/bin/activate
```

### c. 安装与导出依赖

激活环境后，`pip` 会将包安装到当前环境中。
```bash
# 安装单个包
pip install requests

# 根据依赖文件安装
pip install -r requirements.txt

# 将当前环境的依赖导出到文件
pip freeze > requirements.txt
```

### d. 退出环境

执行以下命令即可返回系统全局环境。
```bash
deactivate
```

---

**最佳实践**：始终为每个项目创建独立的虚拟环境，并将环境目录（如 `qwen_env/`）添加到 `.gitignore` 中，避免提交到 Git 仓库。