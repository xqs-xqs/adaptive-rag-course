# Phase 7 — 容器化与压测

- 分支：`refactor/phase-7-deploy-loadtest`
- 前置：Phase 6 已合并
- 本阶段性质：交付「从干净机器一条命令拉起全栈」+ 一份有真实数字的压测报告。压测报告的数字（QPS、P95、瓶颈归因）是整个重构对外输出的浓缩证据。

## 0. 前置假设核对

| # | 假设 | 验证命令 |
|---|---|---|
| A1 | docker compose 可用，Phase 4 的 compose 已含 redis | `docker compose config` |
| A2 | /metrics 与阶段耗时日志可用（压测归因依赖） | `curl -s :8000/metrics \| head` |
| A3 | 评测/冒烟均绿（压测前的健康基线） | `pytest -q && bash scripts/smoke.sh` |

## 1. 涉及的现状文件

- 新增：`Dockerfile`、`.dockerignore`、`app/api/health.py`（/healthz、/readyz）、`loadtest/locustfile.py`、`loadtest/README.md`、`docs/reports/loadtest-report.md`、`tests/contract/test_health.py`
- 修改：`docker-compose.yml`（加 app 服务、healthcheck、依赖顺序）、`app/main.py`（优雅停机钩子、readiness 状态装配）、`requirements-dev`（locust）

## 2. 目标行为

### 2a. 容器化

- 多阶段 Dockerfile：builder 装依赖 → runtime 精简层；**非 root 用户运行**；`.dockerignore` 排除 chroma_db、data、logs、.env、.git
- compose：app + redis；app 配 healthcheck（打 /healthz）；数据目录以 volume 挂载；`docker compose up` 后自动完成索引加载并就绪
- 配置全部经环境变量注入（Phase 1 的 Settings 天然支持），compose 中引用 `.env` 文件

### 2b. 健康探针与优雅停机

- `/healthz`（liveness）：进程活着就 200，不查依赖
- `/readyz`（readiness）：检查 Redis ping、索引已加载（registry 有 active 版本）、熔断器非全开——任一不满足返回 503 + 各项状态 JSON
- 优雅停机：SIGTERM 后停止接收新请求，在超时窗（Settings，默认 15s）内让进行中的 SSE 完成或干净关闭，无 traceback

### 2c. 压测（locust）

- 场景混合：70% 重复问题池（吃缓存）+ 30% 随机变体（穿透缓存）；阶梯并发：5 → 10 → 20 → 40 用户，每级 3 分钟
- 记录每级：QPS、P50/P95/P99、错误率、限流触发数；结合 /metrics 与阶段耗时日志做**瓶颈归因**（预期主导项：LLM/embedding 外呼延迟；验证或推翻它）
- 一组对照实验（至少一个）：如 rerank 开 vs 关、缓存开 vs 关——量化单模块成本，写进报告
- 明确压测对象是本机单实例 + 真实 DashScope API：报告必须注明外部 API 延迟不可控的局限与费用估算，压测前把预计费用报人类批准

## 3. 范围外

- 不上 K8s、不做多实例编排、不接 CDN/网关；不追镜像体积极限
- 不做长稳/浸泡测试（8h+），阶梯短测足够支撑结论

## 4. 实施细节规范

- readiness 的依赖检查带 200ms 超时，探针本身不能被慢依赖拖死
- locustfile 的问题池从评测集抽取（复用真实分布），写死 seed 保证可复现
- 压测期间抓一份 /metrics 快照与日志切片存入 `loadtest/artifacts/`，报告引用原始数据

## 5. 验收标准

| # | 标准 | 命令 | 期望 |
|---|---|---|---|
| C1 | 干净拉起 | 新 clone 目录：`cp .env.example .env`（填 key）→ `docker compose up -d` → `bash scripts/smoke.sh` | 冒烟过，全程无手工步骤 |
| C2 | 非 root | `docker compose exec app whoami` | 非 root 用户名 |
| C3 | readiness 语义正确 | `docker compose stop redis && curl -s -o /dev/null -w "%{http_code}" :8000/readyz`；restart 后再查 | 503（body 指明 redis）→ 200 |
| C4 | liveness 与 readiness 分离 | redis 停止期间 `curl :8000/healthz` | 仍 200 |
| C5 | 优雅停机 | 一条 SSE 进行中 `docker compose stop app`（观察 15s 窗口内） | 流完成或干净关闭；日志无 traceback |
| C6 | 压测报告完整 | 审阅 `docs/reports/loadtest-report.md` | 含：环境说明、各级数字表、瓶颈归因（引用指标/日志证据）、对照实验、局限声明 |
| C7 | 压测中系统行为正确 | 压测原始数据 | 错误率 <1%（限流 429 单列不计错误）；限流按配置触发 |
| C8 | 不变量 | pytest + 评测对比 | 达标 |

## 6. 测试与调试方法

- 容器内起不来先 `docker compose logs app`；再进容器 `docker compose run --rm app python -c "from app.main import app"` 区分依赖问题与代码问题
- 压测数字异常低：先确认不是本机 Docker 资源限额（`docker stats`），再看是否 DashScope 限流（429 计数）
- SSE 在 locust 中的处理：用 catch_response + 流式读取记录首字节时间（TTFT）与总时长两个指标

## 7. 交付物清单

- [ ] PR + 阶段报告
- [ ] docs/reports/loadtest-report.md（数字表 + 归因 + 对照实验）
- [ ] `docs/decisions/0007-deployment.md`：镜像设计、探针语义、压测方法论与局限
- [ ] REFACTOR_PLAN 状态表更新 + 全计划收官备注（各阶段面试考点索引）
