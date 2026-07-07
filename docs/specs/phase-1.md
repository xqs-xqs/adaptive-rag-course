# Phase 1 — 分层与配置

- 分支：`refactor/phase-1-structure-config`
- 前置：Phase 0 已合并（存在 ARCHITECTURE.md 的配置清单与外部调用点清单）
- 本阶段性质：**纯结构性重构，零行为变更**。所有算法逻辑原样搬运，只改"放在哪、怎么装配、怎么取配置"。

## 0. 前置假设核对

| # | 假设 | 验证命令 |
|---|---|---|
| A1 | Phase 0 已合并，基线存在 | `ls docs/baseline/ && git log --oneline -5 main` |
| A2 | ARCHITECTURE.md 配置清单存在且非空 | `grep -A3 "配置清单" docs/ARCHITECTURE.md \| head` |
| A3 | 外部调用点清单存在 | `grep -A3 "外部调用" docs/ARCHITECTURE.md \| head` |
| A4 | 冒烟可通过（重构前最后确认） | `bash scripts/smoke.sh; echo $?` |

## 1. 涉及的现状文件

- 移动/改写：根目录全部业务 `.py`（app / config / chunking / indexing / retrieval / generation / txt_parser / evaluation）
- 新增：`app/` 包（结构见下）、`.env.example`、`scripts/run_eval.py`、`scripts/ingest.py`（若现状有独立灌库入口则搬运，无则暂缓）
- 修改：`requirements.txt`（新增 pydantic-settings）、`.gitignore`（确保 `.env`、`chroma_db/` 被忽略）、`scripts/smoke.sh`（更新启动命令）

## 2. 目标行为

### 2a. 目标目录结构（**钉死。后续所有阶段的文件引用以此为准**）

```
app/
  main.py                 # FastAPI 应用装配 + 手工依赖注入（唯一知道具体实现类的地方）
  api/
    routes.py             # HTTP / SSE 端点，只调用 services，不 import infra
  core/
    config.py             # pydantic-settings 的 Settings 类，全部配置的唯一来源
  domain/
    chunking.py           # 分块逻辑（纯函数为主）
    parsing.py            # 原 txt_parser
    fusion.py             # RRF 融合、diversity 过滤（从 retrieval 中拆出的纯计算）
  services/
    retrieval_service.py  # 编排：路由→混合检索→融合→重排
    generation_service.py # 编排：prompt 组装→LLM→会话状态
    indexing_service.py   # 编排：解析→分块→建索引
  infra/
    llm.py                # LLMClient 抽象基类 + DashScopeLLM 实现
    embeddings.py         # EmbeddingClient 抽象基类 + DashScopeEmbedding 实现
    vectorstore.py        # VectorStore 抽象基类 + ChromaVectorStore 实现
    reranker.py           # Reranker 抽象基类 + 现有实现
    lexical.py            # BM25 索引封装（jieba 分词在此）
scripts/
  run_eval.py             # 评测入口（原 evaluation.py 的 CLI 壳，评测逻辑可留 scripts/ 或 evaluation/ 下）
  smoke.sh
tests/                    # 本阶段只建目录 + 空 __init__，Phase 2 填充
```

### 2b. 接口与装配规范

- 四个抽象基类用 `abc.ABC + @abstractmethod`，方法签名从现有调用方式反推，保持最小（不要设计"将来可能用到"的方法）
- `services/` 与 `api/` 中不得出现 `dashscope`、`chromadb` 的直接 import——只依赖 infra 的抽象基类
- 装配只发生在 `app/main.py`：构造具体实现 → 注入 services → 挂载路由。不引入任何 DI 框架
- `evaluation` 的逻辑也改为通过接口取依赖，保证 Phase 2 能替换成 fake

### 2c. 配置规范

- `Settings(BaseSettings)` 收编 ARCHITECTURE.md 配置清单中的**全部**条目，每个字段带类型和默认值；API key 类字段**无默认值**
- 缺少 `DASHSCOPE_API_KEY` 时：启动必须立即失败，错误信息明确指出缺哪个变量（fail fast）
- 提交 `.env.example`（含全部变量名 + 假值）；确认 `.env` 在 .gitignore 中
- 代码中不得残留任何形如 `"your-api-key-here"` 的默认密钥字符串

## 3. 范围外（明确不做）

- 不改任何算法逻辑、不加超时重试（Phase 3）、不加缓存（Phase 4）、不改日志（Phase 5）
- 不写单元测试（Phase 2），本阶段的安全网是冒烟 + 评测对比
- 发现原逻辑的 bug → 记 BACKLOG，不修（导入错误、路径错误等纯搬运破损除外）

## 4. 实施细节规范

- **搬运优先于重写**：函数体尽量原样移动；确需改动（如参数从读全局配置改为显式传入）在 commit message 中说明
- 建议 commit 序列：建骨架 → 迁 domain → 迁 infra（一个文件一个 commit）→ 迁 services → 迁 api/main → 接 Settings → 删除根目录旧文件 → 更新 smoke/eval 入口
- **中间态可以碎，每个 commit 必须完整**：`python -c "from app.main import app"` 在每个 commit 上都要能过
- 循环导入是本阶段最常见的坑：依赖方向强制为 api → services → (domain, infra)，domain 不 import 任何本项目模块

## 5. 验收标准

| # | 标准 | 命令 | 期望 |
|---|---|---|---|
| C1 | 根目录无业务 py | `ls *.py 2>/dev/null` | 无输出 |
| C2 | 分层隔离成立 | `grep -rn "dashscope\|chromadb" app/services/ app/api/ app/domain/` | 无输出 |
| C3 | 无默认密钥 | `grep -rn "your-api-key\|sk-" --include="*.py" .` | 无输出 |
| C4 | fail fast | `env -u DASHSCOPE_API_KEY python -c "from app.main import app"` | 非零退出 + 明确报错含变量名 |
| C5 | 配置收编完整 | 对照 ARCHITECTURE.md 配置清单逐条检查 Settings 字段 | 无遗漏（报告中列对照表） |
| C6 | 冒烟通过 | `bash scripts/smoke.sh; echo $?` | 0 |
| C7 | 行为未变（关键门禁） | 全量评测 → 对比基线 | 检索侧指标差值 ≤ I1 阈值；生成侧记录即可 |
| C8 | 每个 commit 可导入 | `git rebase --exec 'python -c "from app.main import app"' main`（或逐 commit checkout 验证） | 全过 |

## 6. 测试与调试方法

- 循环导入报错：`python -X importtime -c "from app.main import app" 2>&1 | tail -30` 定位；解法优先"下沉共享类型到 domain"，禁止函数内 import 糊弄
- 评测指标意外下跌：先 diff 检索链路相关文件确认是否搬运时改了默认参数（最常见：top_k、阈值在搬进 Settings 时抄错默认值——逐个对照 ARCHITECTURE.md 清单里的"当前值"）
- smoke 失败先看 uvicorn 启动日志，再 `curl -v` 看具体断在哪层

## 7. 交付物清单

- [ ] PR + 阶段报告（含 C1–C8 原样输出、配置对照表）
- [ ] `docs/decisions/0001-layered-architecture.md`：一页 ADR——为什么这样分层、备选（保持平铺 / 按技术分层 vs 按领域分层）、取舍
- [ ] REFACTOR_PLAN 状态表更新
