# Phase 6 — 索引生命周期

- 分支：`refactor/phase-6-index-lifecycle`
- 前置：Phase 5 已合并（切换过程需要日志与指标佐证）
- 本阶段性质：回答面试必问题「线上文档更新了怎么办，停服重建吗？」——增量摄取、版本化索引、零停机热切换、双索引一致性。

## 0. 前置假设核对

| # | 假设 | 验证命令 |
|---|---|---|
| A1 | 当前摄取为全量重建（无增量机制） | 阅读 `app/services/indexing_service.py` + `scripts/ingest.py`，报告中引用证据行号 |
| A2 | BM25 索引在进程内存、启动时构建 | `grep -rn "BM25\|bm25" app/infra/lexical.py` |
| A3 | Chroma 支持按 id 删除与多 collection | `python -c "import chromadb; print(chromadb.__version__)"`，对照该版本 API |
| A4 | 检索缓存 key 已含 index_version 占位 | `grep -rn "index_version" app/` |

## 1. 涉及的现状文件

- 新增：`app/services/index_manifest.py`（清单与差量计算）、`app/core/index_registry.py`（活动版本指针 + 原子切换）、`tests/unit/test_manifest.py`、`tests/unit/test_registry.py`、`tests/contract/test_hot_swap.py`、`scripts/swap_drill.py`（切换演练脚本）
- 修改：`app/services/indexing_service.py`（增量摄取）、`app/infra/vectorstore.py`（版本化 collection、按 id 删除）、`app/infra/lexical.py`（从规范化 chunk 存储重建）、`app/services/retrieval_service.py`（经 registry 取当前索引对、缓存 key 用真实版本号）、`app/api/routes.py`（`/admin/reindex` 与 `/admin/swap` 或等价机制）

## 2. 目标行为

### 2a. 单一事实源与清单

- 规范化 chunk 存储：摄取产物统一落 `data/chunks/{version}/chunks.jsonl`（id、text、metadata、parent 关系）——**BM25 与向量索引都只从它构建**，这是双索引一致性的根
- `data/manifest.json`：每个源文档 → sha256、chunk id 列表、摄取时间。增量逻辑：新/变更文档才重新解析分块与 embedding；删除的文档其 chunk 从下一版本剔除

### 2b. 版本化与热切换

- 向量侧：collection 命名 `courses_{version}`；BM25 侧：按版本从 chunks.jsonl 构建内存索引
- `IndexRegistry` 持有 `active: IndexPair(version, vectorstore_handle, bm25_handle)`；切换 = 构建新 IndexPair 完成后**原子替换指针**（单赋值 + 锁保护读取），保证任一请求看到的向量/BM25 属于同一版本，**绝不混版本**
- 响应 meta 增加 `index_version`（契约变更，声明并更新契约测试）；检索缓存 key 的 index_version 接入真实值——切版本天然全量失效，无需手动清缓存（这个设计在 ADR 里点名，是 Phase 4 埋的钩子兑现）
- 保留上一版本用于回滚：`/admin/swap --to {version}` 可切回；旧版本清理策略（保留最近 2 个）写入 Settings

### 2c. 摄取的可观测

- 摄取过程输出结构化日志：处理/跳过/删除的文档数、新增 chunk 数、embedding 调用数、耗时——增量是否生效由这些数字证明

## 3. 范围外

- 不做定时调度/文件监听自动重建（手动触发端点即可，ADR 记录演进方向）
- 不迁移 Chroma 到 server 模式或 Milvus（记 BACKLOG 作对比方案素材）
- BM25 按版本全量重建（语料量级下成本可忽略，ADR 中给出量级论证），不做 BM25 增量更新

## 4. 实施细节规范

- manifest 写入必须原子（写临时文件 + rename）
- 摄取与服务同进程运行时，构建新索引期间旧索引照常服务；embedding 复用 Phase 4 的缓存层（变更文档的未变 chunk 命中缓存——报告中用调用计数证明）
- `/admin/*` 端点加简单鉴权（Settings 中的 admin token，header 校验），README 注明生产应网络隔离

## 5. 验收标准

| # | 标准 | 命令 | 期望 |
|---|---|---|---|
| C1 | 增量：只处理变更 | 修改 1 个源文档 → 触发 reindex → 看摄取日志 | 仅该文档被解析；embedding 调用数 == 该文档新 chunk 数（缓存命中另计并展示） |
| C2 | 删除生效 | 删 1 个源文档 → reindex + swap → 用只有它能答的问题查询 | 新版本检索不到其 chunk（golden 用例断言） |
| C3 | 双索引一致 | 构建后自检：BM25 文档数 == 该版本 chunks.jsonl 行数 == Chroma collection count | 三者相等，自检输出入日志 |
| C4 | 热切换零中断 | `python scripts/swap_drill.py`：循环请求（10 rps 持续 60s）期间执行 swap | 0 个非 2xx；meta 的 index_version 单调切换、无交错混版 |
| C5 | 回滚可用 | swap 到旧版本 → 冒烟 | 通过；meta 显示旧版本号 |
| C6 | 缓存随版本失效 | swap 后重复此前已缓存的查询 | 首次 miss（日志证明），返回新版本结果 |
| C7 | 不变量 | pytest + 冒烟 + 评测对比（评测跑在最新版本索引上） | 达标 |

## 6. 测试与调试方法

- 单元层：manifest 差量计算用临时目录 + 构造文件三态（新增/修改/删除）覆盖；registry 原子性用多线程读 + 单线程换指针压测断言"读到的 pair 永远同版本"
- swap_drill.py 同时充当演示脚本：输出切换时间点、期间成功率、版本序列
- 排查混版：日志按 trace_id 关联"检索用的 version"与"meta 返回的 version"是否一致

## 7. 交付物清单

- [ ] PR + 阶段报告（含 C4 演练输出全文）
- [ ] `docs/decisions/0006-index-lifecycle.md`：单一事实源设计、为何指针切换而非双写、回滚策略、BM25 全量重建的量级论证
- [ ] REFACTOR_PLAN 状态表更新
