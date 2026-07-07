# Phase 0 — 系统解剖 + 基线 + 热身修复

- 分支：`refactor/phase-0-baseline`
- 前置：无（这是第一个阶段）
- 本阶段性质：**只读为主 + 两个微小修复**。目的：建立全局地图、锁定可对比的基线、验证整个协作流程能跑通。

## 0. 前置假设核对（逐条验证，任何一条不符 → 停止并报告）

| # | 假设 | 验证命令 |
|---|---|---|
| A1 | 仓库根目录平铺存在核心模块：app.py / config.py / chunking.py / indexing.py / retrieval.py / generation.py / evaluation.py / txt_parser.py | `ls *.py` |
| A2 | config.py 中存在硬编码模型名，且可能有 `qwen3.6-plus` 这个疑似 typo | `grep -n "qwen" config.py` |
| A3 | retrieval.py 的 diversity 过滤中存在两分支同值的疑似 bug（形如 `max_per = 2 if ... else 2`） | `grep -n "max_per" retrieval.py` |
| A4 | 评测入口是 evaluation.py，支持某种模式参数 | `grep -n "argparse\|--mode\|add_argument" evaluation.py` |
| A5 | 向量库为 ChromaDB 本地持久化 | `grep -rn "chroma" --include="*.py" -i . \| head -20` |
| A6 | 仓库能在本地装起依赖并启动（人类环境已有 DASHSCOPE_API_KEY） | `pip install -r requirements.txt` 后启动服务 |

> A2 / A3 若已被修复（grep 无结果），不算偏差——在报告中注明「已不存在」，跳过对应修复任务即可。其余假设不符必须停。

## 1. 涉及的现状文件

- 只读全仓库所有 `.py`、`requirements.txt`、`README.md`
- 允许修改：`config.py`（若 A2 成立）、`retrieval.py`（若 A3 成立）
- 允许新增：`docs/ARCHITECTURE.md`、`docs/BACKLOG.md`、`docs/baseline/`、`docs/reports/phase-0-report.md`、`scripts/smoke.sh`
- 本 kit 自带的 `CLAUDE.md`、`docs/REFACTOR_PLAN.md`、`docs/specs/`、模板、PR 模板：随本阶段第一个 commit 入库

## 2. 目标行为

1. **产出 `docs/ARCHITECTURE.md`**，必须包含以下四节，且全部基于真实代码（禁止臆测）：
   - 文件清单：每个 .py 一行——职责、被谁 import、import 了谁
   - 查询生命周期：一次 HTTP 请求从进入到 SSE 流式返回的完整数据流，按顺序列出经过的函数（`文件名:函数名`），标注每一步的输入/输出形态
   - 配置清单：所有硬编码常量（模型名、top_k、阈值、路径等）——位置、当前值、一句话作用。**这是 Phase 1 的直接输入**
   - 外部调用点清单：所有产生网络 I/O 的调用（embedding、LLM、rerank 等）——位置、同步还是异步、有无超时/重试。**这是 Phase 3 的直接输入**
2. **修复两个已知 bug**（若仍存在）：A2 的模型名 typo、A3 的同值分支。各自独立 commit，`fix(...)` 前缀。修复 A3 前先在报告里写清你判断的正确意图（两分支本应是什么值）。
3. **锁定基线**：
   - 全量评测跑**两遍**（跑之前把预计费用/耗时报给人类确认），结果存 `docs/baseline/eval-run1.json`、`eval-run2.json`
   - 把两次指标、差值填入 REFACTOR_PLAN 第 6、7 节，据两次差值提议 I1 阈值校准值
   - `pip freeze > docs/baseline/requirements.lock.txt`
4. **创建 `scripts/smoke.sh`**：启动服务（后台）→ 用 curl 发 1 个固定问题 → 校验 HTTP 200 且响应含预期关键词 → 关闭服务。问题和关键词与人类商定后写死在脚本里。
5. **创建 `docs/BACKLOG.md`**：本阶段读代码过程中发现的所有问题（除 A2/A3 外一律不修），按模板逐条记录。

## 3. 范围外（明确不做）

- 不移动任何文件、不改任何目录结构（那是 Phase 1）
- 不修 BACKLOG 里的任何其他 bug、不做任何"顺手优化"
- 不动 adaptive-rag-mentor/ 技能目录

## 4. 实施细节规范

- ARCHITECTURE.md 的数据流一节，每个步骤必须能用 `grep -n` 在代码中定位到，人类会抽查
- 评测两次运行之间不要改任何代码/配置，保证差值反映的是纯随机性
- commit 顺序建议：`docs: add refactor kit` → `fix(config): ...` → `fix(retrieval): ...` → `docs: add architecture map` → `test: add smoke script` → `docs: record baseline`

## 5. 验收标准（逐条执行，输出贴入阶段报告）

| # | 标准 | 命令 | 期望 |
|---|---|---|---|
| C1 | typo 已清除 | `grep -rn "qwen3.6-plus" .` | 无输出（或报告注明本就不存在） |
| C2 | 同值分支已修复 | `grep -n "max_per" retrieval.py` | 两分支值不同，或报告说明为何维持 |
| C3 | 架构文档完整 | `grep -c "^## " docs/ARCHITECTURE.md` | ≥ 4（四个必需章节） |
| C4 | 每个 .py 入册 | 对比 `ls *.py` 与文件清单 | 无遗漏 |
| C5 | 基线锁定 | `ls docs/baseline/` | 两个 eval JSON + requirements.lock.txt |
| C6 | 冒烟通过 | `bash scripts/smoke.sh; echo $?` | 输出 0 |
| C7 | 计划表已更新 | 查看 REFACTOR_PLAN 第 4/6/7 节 | 状态、基线、阈值提案已填 |

## 6. 测试与调试方法

- 服务起不来：先 `python -c "import app"` 看导入错误，再查 `.env` / 环境变量是否就位
- 评测中途失败：检查是否 DashScope 限流（429）——若是，串行重跑并在报告记录，不要改评测代码
- smoke.sh 里用 `curl -N` 处理 SSE；用 `timeout 60` 包裹防止挂死

## 7. 交付物清单

- [ ] PR：`refactor/phase-0-baseline` → main
- [ ] docs/ARCHITECTURE.md
- [ ] docs/baseline/（2 个 JSON + lock 文件）
- [ ] scripts/smoke.sh
- [ ] docs/BACKLOG.md（含发现的问题）
- [ ] docs/reports/phase-0-report.md（含全部验收命令原样输出 + I1 阈值校准提案）
- [ ] REFACTOR_PLAN 状态表更新

## 8. 人类侧任务（审查课前完成）

- 依据 ARCHITECTURE.md，不看代码向 Claude（Claude.ai）复述一遍查询数据流，接受挑错——这是本阶段人类的验收标准
