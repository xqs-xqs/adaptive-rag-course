# Phase 5 — 可观测性

- 分支：`refactor/phase-5-observability`
- 前置：Phase 4 已合并（缓存计数器等埋点对象已存在）
- 本阶段性质：让每个请求可追踪、每个阶段耗时可量化、成本可核算。产出物是面试时那句话的证据：「P95 x.x 秒，瓶颈在 rerank，缓存命中后降到 x.x」。

## 0. 前置假设核对

| # | 假设 | 验证命令 |
|---|---|---|
| A1 | 当前日志为非结构化 logging/print 混用 | `grep -rn "print(\|logging\." app/ \| head -20` |
| A2 | 请求主链路：路由→检索→重排→生成，各步骤在 services 中有清晰函数边界 | 对照 ARCHITECTURE.md 数据流一节 |
| A3 | 缓存命中/错误计数已按 Phase 4 落日志 | `grep -rn "cache_hit\|cache_miss" app/` |

## 1. 涉及的现状文件

- 新增：`app/core/logging.py`（structlog 配置）、`app/core/observability.py`（trace_id contextvar、计时器、Prometheus 指标定义）、`app/api/middleware.py`（请求中间件）、`scripts/analyze_logs.py`、`tests/unit/test_observability.py`
- 修改：`app/main.py`（挂中间件、/metrics 端点）、各 services（阶段计时 + 结构化日志改造）、`app/infra/llm.py`（token 用量与成本记录）、`requirements.txt`（structlog、prometheus-client）

## 2. 目标行为

### 2a. 结构化日志

- structlog 输出 JSON 行；全项目替换 print 与裸 logging
- 中间件为每个请求生成 `trace_id`（优先取入站 `X-Request-ID`），通过 contextvar 注入，**该请求生命周期内所有日志行自动携带**——包括线程池中执行的部分（注意 contextvar 跨 to_thread 的传播，structlog 的 contextvars 集成 + anyio 默认传播，需测试验证而非假设）
- 统一字段规范：`ts, level, event, trace_id, dep, duration_ms` + 事件私有字段；禁止 f-string 拼日志正文，全部走键值对

### 2b. 阶段计时与成本

- 每请求记录：`t_route / t_retrieve / t_rerank / t_generate / t_total`（ms）——日志一条汇总行 + 写入 SSE 结束事件的 meta（契约变更，显式声明并更新契约测试）
- LLM 调用记录 prompt/completion token 数与按价目估算的成本（单价入 Settings，标注"估算"）

### 2c. 指标端点

- prometheus-client：`/metrics` 暴露——Counter：请求总数（按端点/状态码）、缓存命中/未命中、降级事件（按依赖）、重试次数；Histogram：各阶段耗时、总耗时
- 不部署 Prometheus/Grafana（非目标）；`scripts/analyze_logs.py` 从 JSONL 日志离线算 P50/P95/P99 分阶段表格——这是拿"面试数字"的工具

## 3. 范围外

- 不接 OpenTelemetry 全链路（单服务没有跨服务传播需求，写进 ADR 作为"何时才需要"的答案）
- 不上 Grafana/Loki 等平台件；不做采样（量级不需要）

## 4. 实施细节规范

- 计时器用 `time.perf_counter()`；封装 context manager `stage_timer("retrieve")`，兼顾同步/异步
- /metrics 不鉴权但在 README 注明生产需隔离（面试考点）
- 日志量控制：检索候选列表等大对象只记 id 与数量，禁止整段 chunk 文本入日志（既省空间也防敏感内容落盘）

## 5. 验收标准

| # | 标准 | 命令 | 期望 |
|---|---|---|---|
| C1 | 日志是合法 JSON | 启动后请求一次：`tail -20 log \| jq -e . >/dev/null; echo $?` | 0 |
| C2 | trace_id 贯穿 | 测试捕获一次请求全部日志，断言所有行 trace_id 相同；含线程池内日志行 | 过 |
| C3 | 无 print 残留 | `grep -rn "print(" app/` | 无输出（scripts/ 除外） |
| C4 | 阶段耗时齐全 | 一次请求的汇总日志行 | 五个 t_* 字段均为正数；SSE meta 含同款 |
| C5 | /metrics 可用且会动 | `curl -s :8000/metrics \| grep -E "rag_requests_total\|rag_stage_seconds"`，请求 3 次后再看 | 指标存在且计数递增 |
| C6 | 分位数报表 | 打 20 个混合请求 → `python scripts/analyze_logs.py logs/app.jsonl` | 输出各阶段 P50/P95 表 |
| C7 | token/成本入账 | 汇总日志行含 prompt_tokens / completion_tokens / est_cost | 数值合理（与 DashScope 控制台抽查一致，人工核对入报告） |
| C8 | 不变量 | pytest + 冒烟 + 评测对比 | 达标（契约测试已按 meta 扩展更新） |

## 6. 测试与调试方法

- contextvar 丢失（线程池内日志缺 trace_id）是本阶段第一大坑：写一个专门测试覆盖 to_thread 路径；若丢失，显式在提交给线程池的闭包里绑定 structlog contextvars
- jq 快速排查：`jq 'select(.trace_id=="xxx")' logs/app.jsonl` 还原单请求全过程
- Histogram 桶不合理导致 P95 失真：先用 analyze_logs.py 的精确分位数校准桶边界再定型

## 7. 交付物清单

- [ ] PR + 阶段报告（附一张真实的 P50/P95 分阶段表——这张表直接进面试材料）
- [ ] `docs/decisions/0005-observability.md`：为什么 structlog+prometheus 而非 OTel 全家桶、日志字段规范、成本估算口径
- [ ] REFACTOR_PLAN 状态表更新
