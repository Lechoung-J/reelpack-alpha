# InternVL3.5-4B — 可选 Ollama 视觉模型

默认 OCR 使用 macOS Vision，摘要与校正默认不运行。需要 Ollama 视觉后端时可自行准备此模型；模型权重及其许可独立于本项目源码。

## 快速安装

```bash
# 1. 下载 GGUF (bartowski Q4_K_M 量化, 2.7GB)
hf download bartowski/OpenGVLab_InternVL3_5-4B-GGUF \
  --include "*Q4_K_M*" \
  --local-dir ~/models/internvl3.5-4b/

# 2. 创建 Ollama Modelfile
cat > /tmp/Modelfile.internvl <<EOF
FROM $HOME/models/internvl3.5-4b/OpenGVLab_InternVL3_5-4B-Q4_K_M.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.1
PARAMETER num_ctx 8192
EOF

# 3. 导入 Ollama
ollama create internvl3.5:4b -f /tmp/Modelfile.internvl

# 4. 验证
ollama list | grep internvl
```

## 使用

```bash
# 本地文件转写，显式启用可选 Ollama OCR
.venv/bin/python Core/core.py --input ~/video.mp4 --ocr-backend ollama --vision-model internvl3.5:4b
```
