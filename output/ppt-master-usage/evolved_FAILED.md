---
name: ppt-master-usage
description: >
  使用 ppt-master 生成原生可编辑 PPTX 的完整流程。记录了正确的 Python 路径、依赖安装、和项目初始化方法。
tags: ["ppt", "powerpoint", "automation"]
category: productivity
---

# ppt-master 使用指南

## 环境要求
- Python 3.10+
- ppt-master 安装在 `~/.hermes/skills/ppt-master/`
- **关键**：必须用 hermes-agent 的 venv Python，不能用系统 python3

## 正确 Python 路径
```bash
~/.hermes/hermes-agent/.venv/bin/python3
```

## 依赖安装（一次性）
所有依赖通过 uv pip 安装（因为 hermes-agent venv 没有 pip 模块）：
```bash
uv pip install svglib==1.5.1 reportlab PyMuPDF mammoth markdownify ebooklib nbconvert requests beautifulsoup4 pillow numpy cairosvg curl_cffi google-genai openai python-pptx cssselect2==0.9.0
```

注意：
- `svglib<1.6`（1.5.1 是已知稳定版本）
- `google.genai` 命名空间导入：`import google.genai as genai`（不是 `import genai`）
- cairosvg 需要系统有 cairo 库，Linux 上通常已有

## 项目初始化
```bash
SKILL_DIR=~/.hermes/skills/ppt-master/skills/ppt-master
~/.hermes/hermes-agent/.venv/bin/python3 ${SKILL_DIR}/scripts/project_manager.py init 项目名称 --format ppt169
```

## PPT 生成流程（AI 驱动）
1. 用户提供材料（PDF/DOCX/Markdown/纯文字均可）
2. 初始化项目：`project_manager.py init`
3. 选择模板（用户提供内容后，AI 确认设计规范）
4. Strategist 生成 `design_spec.md`
5. Executor 逐页生成 SVG
6. Post-processing：`finalize_svg.py` → `svg_to_pptx.py`

## 快速上手（不需 AI 介入的最小路径）
如果用户直接给内容说"做PPT"，流程是：
1. `project_manager.py init <name> --format ppt169`
2. AI 读取内容 → 生成 design_spec.md → 确认
3. AI 生成 SVG → 导出 PPTX

## 导出流程：finalize_svg.py 是关键步骤！

**血的教训**：如果跳过 `finalize_svg.py`，直接导出 `svg_output/`，生成的 PPTX 每一页都是**图片回退**（不可编辑的截图），文字、颜色、布局完全无法修改。

**正确流程**：
```bash
# Step 1: SVG 后处理（必须！生成 svg_final/）
~/.hermes/hermes-agent/.venv/bin/python3 ${SKILL_DIR}/scripts/finalize_svg.py -i svg_output/ -o svg_final/

# Step 2: 从 svg_final/ 导出（不是 svg_output/！）
~/.hermes/hermes-agent/.venv/bin/python3 ${SKILL_DIR}/scripts/svg_to_pptx.py -s final -i svg_final/
```

**为什么**：`svg_output/` 里的是生 SVG（含 AI 生成文字路径），`svg_final/` 是后处理过的矢量形状（DrawingML 原生形状）。只有 `svg_final/` 导出的 PPTX 才能直接编辑文字和布局。

## 备选工具：PPTAgent（中科院）

如果 ppt-master 效果不够好，可以探索 **PPTAgent**：
- GitHub: https://github.com/icip-cas/PPTAgent
- EMNLP 2025 论文，4.1k Stars
- 思路：两阶段（分析参考PPT → 迭代生成新PPT）+ PPTEval 三维评估
- V2 版集成了 Deep Research + 文生图 + Agent 环境
- **支持 OpenClaw 集成**（`uvx pptagent onboard` 配置）
- 安装：`uvx pptagent onboard` 即可
- 生成：`uvx pptagent generate "主题" -o output.pptx`
- 适合：需要更高设计质量的场景，或作为 ppt-master 的对比参考

## 常见问题
- 导入 `google.genai` 失败 → 用 `import google.genai as genai`
- 找不到脚本 → 检查 `${SKILL_DIR}/scripts/` 目录结构
- 依赖报错 → 确认用的是 hermes-agent venv 的 python，不是系统 python3
- 生成的 PPTX 文字无法编辑 → 一定是跳过了 `finalize_svg.py`，重新从 Step 1 开始
