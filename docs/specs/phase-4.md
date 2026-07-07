# Phase 4 — 缓存与无状态化

- 分支：`refactor/phase-4-cache-stateless`
- 前置：Phase 3 已合并（缓存必须 fail-open，依赖 Phase 3 的依赖不可用异常体系）
- 本阶段性质：引入 Redis 三层缓存 + 会话状态外置 + 事件循环阻塞审计。与黑马点评的 Redis 知识（击穿/穿透/雪崩）直接互认。

## 0. 前置假设核对

| # | 假设 | 验证命令 |
|---|---|---|
| A1 | 本机可跑 Docker（Redis 用容器起） | `docker --version && docker compose version` |
| A2 | 会话状态当前在进程内存（字典/列表持有对话历史） | `grep -rn "history\|session\|conversation" app/services/generation_service.py \| head` |
| A3 | DashScope SDK 调用为同步阻塞（事件循环审计的前提） | `grep -rn "async def\|await" app/infra/ \| head`，对照调用方式判断 |
| A4 | 降级异常体系可用 | `grep -n "DependencyUnavailable" app/core/resilience.py` |

> A2/A3 若与假设相反（例如已是异步或已无状态），对应子任务在报告中标记「不适用 + 证据」，不算偏差。

## 1. 涉及的现状文件

- 新增：`app/infra/cache.py`（CacheClient 抽象 + RedisCache + InMemoryCache）、`docker-compose.yml`（redis 服务）、`tests/unit/test_cache.py`、`tests/unit/test_session_store.py`、`tests/contract/test_stateless.py`、`scripts/latency_probe.py`
- 修改：`app/infra/embeddings.py`（embedding 缓存）、`app/services/retrieval_service.py`（检索结果缓存 + 击穿锁）、`app/services/generation_service.py`（会话外置 + 可选语义缓存）、`app/core/config.py`、`app/main.py`（装配）、`requirements-dev` 加 fakeredis

## 2. 目标行为

### 2a. 三层缓存（key 规范固定如下，不得自行发明格式）

| 层 | key | value | TTL | 说明 |
|---|---|---|---|---|
| embedding | `emb:{model}:{sha256(text)}` | 向量（json/bytes） | 7d | 省钱主力；命中则完全不出网 |
| 检索结果 | `ret:{index_version}:{sha256(norm_query + 检索参数指纹)}` | chunk id 列表+分数 | 1h ± 抖动 | `index_version` 本阶段先读 Settings 常量 `"v1"`，为 Phase 6 预留失效钩子 |
| 答案语义缓存 | `ans:{index_version}:{向量近邻}` | 完整答案 | 24h | **默认关闭**（Settings 开关），相似度阈值 ≥ 0.95 起步；命中时响应 meta 标注 `cached: true` |

- 三防齐备并写测试：**击穿**——同 key 并发 miss 时用 `SET NX` 锁 + 等待重查，只放一个请求回源；**穿透**——空结果也缓存（短 TTL 60s）；**雪崩**——TTL 加随机抖动（±10%）
- **fail-open 是硬约束**：Redis 不可用时所有缓存层静默旁路（WARN 日志 + degraded 标记），业务照常。缓存永远不能成为新的故障点

### 2b. 会话状态外置

- 会话历史移入 Redis：`sess:{session_id}` → 消息列表，TTL 24h，长度上限（Settings，如 20 轮）
- 服务进程内不再持有任何跨请求状态；`session_id` 由客户端携带（沿用现有机制，若现状无 session 机制则在执行计划中提出最小方案获批）

### 2c. 事件循环阻塞审计（交付一份审计表，进报告）

- 列出所有 async 路径中调用同步阻塞函数的位置（重点：dashscope SDK、Chroma 查询、CPU 重的 rerank）
- 修复：阻塞调用包进 `anyio.to_thread.run_sync`（或 `run_in_executor`），线程池大小入 Settings
- 修复的正确性由 C5 的并发探针验证

## 3. 范围外

- 不做分布式锁的 Redlock、不做多级本地缓存（Caffeine 类比）、不上 Redis Cluster
- 不把 BM25/向量索引搬进 Redis（Phase 6 处理索引问题）
- 语义缓存默认关（风险：相似但约束不同的问题命中错误答案——这个坑本身是面试考点，写进 ADR）

## 4. 实施细节规范

- CacheClient 接口最小化：`get / set / delete / setnx`（带 TTL 参数），序列化在调用方，接口只管 bytes/str
- 缓存读写全部 try/except 包裹并计数（命中/未命中/错误三个计数器，Phase 5 会接指标，本阶段先日志）
- 单元测试用 fakeredis；击穿测试用两个并发任务 + 可控的慢回源 fake，断言回源函数只被调用一次

## 5. 验收标准

| # | 标准 | 命令 | 期望 |
|---|---|---|---|
| C1 | 缓存行为正确 | `pytest tests/unit/test_cache.py -q` | 命中/未命中/TTL/三防各有用例且过 |
| C2 | fail-open | 契约测试：注入不可用的 cache fake → 请求照常 200 | 过；日志含缓存旁路 WARN |
| C3 | 无状态验证 | `pytest tests/contract/test_stateless.py -q`；手动：多轮对话中途重启服务，第二轮仍有上下文 | 过；手动过程录入报告 |
| C4 | 进程内状态清零 | `grep -rn "self\.history\|_sessions\|conversation_store" app/services/`（按 A2 实际命名调整） | 无跨请求内存状态 |
| C5 | 事件循环不再被阻塞 | 测试：fake LLM 同步 sleep 2s，期间并发请求 `/健康探针或轻端点` | 轻请求响应 < 100ms（修复前应 >2s，修复前后对照写入报告） |
| C6 | 缓存收益可量化 | `python scripts/latency_probe.py`（同一问题连打 10 次，输出 P50） | 命中后 P50 较首个请求改善 ≥ 50%（待校准），embedding 调用计数不增长 |
| C7 | Redis 宕机演练 | `docker compose stop redis && bash scripts/smoke.sh` | 冒烟仍过（降级日志出现）；`start` 后缓存恢复 |
| C8 | 不变量 | pytest + 冒烟 + 评测对比 | 达标 |

## 6. 测试与调试方法

- Redis 侧观察：`docker compose exec redis redis-cli` → `KEYS 'emb:*' | head`（仅 dev 用 KEYS）、`TTL <key>`、`MONITOR` 看实时命令流
- 击穿锁调试：MONITOR 里应看到并发 miss 时只有一个 `SET ... NX` 成功
- 语义缓存误命中排查：日志记录命中时的相似度分值与命中 key，人工抽查 5 条

## 7. 交付物清单

- [ ] PR + 阶段报告（含阻塞审计表、C5 前后对照、C6 数据）
- [ ] `docs/decisions/0004-caching.md`：key 设计、TTL 取值、三防实现、语义缓存为何默认关、与黑马点评知识点的对照表
- [ ] REFACTOR_PLAN 状态表更新
