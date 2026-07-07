# Phase 3 — 容错、限流、降级

- 分支：`refactor/phase-3-resilience`
- 前置：Phase 2 已合并（改动受测试保护；fake 可用于故障注入）
- 本阶段性质：让"任何一个外部依赖挂掉"都有确定的、被测试证明的系统行为。这是 Java 后端面试权重最高的一块，实现要**手写而非引重型库**，因为目的是人类能逐行讲清。

## 0. 前置假设核对

| # | 假设 | 验证命令 |
|---|---|---|
| A1 | Phase 0 的外部调用点清单存在 | `grep -A5 "外部调用" docs/ARCHITECTURE.md` |
| A2 | 所有外部调用已收拢进 app/infra/ | `grep -rn "dashscope" app/ --include="*.py" -l`（应只出现在 infra） |
| A3 | 契约测试存在（降级不能破坏契约） | `pytest tests/contract/ -q` |

## 1. 涉及的现状文件

- 新增：`app/core/resilience.py`（重试 + 熔断器）、`app/api/ratelimit.py`（令牌桶中间件）、`tests/unit/test_resilience.py`、`tests/unit/test_ratelimit.py`、`tests/contract/test_degradation.py`
- 修改：`app/infra/llm.py` / `embeddings.py` / `reranker.py`（接入超时/重试/熔断）、`app/services/retrieval_service.py` / `generation_service.py`（降级链路）、`app/core/config.py`（全部容错参数入 Settings）、`app/main.py`（装配 + 故障注入开关）

## 2. 目标行为

### 2a. 出站调用防护（对外部调用点清单中的每一个调用）

- 显式超时：连接超时与读超时分开配置（Settings：`llm_timeout_s` 等，按依赖分组）
- 重试：指数退避 + 抖动，最多 2 次重试；**仅对可重试错误**（超时、429、5xx）；4xx（除 429）一律不重试。所有调用均为读操作，天然幂等——这一点写进 ADR
- 熔断器：**手写**，三态（closed/open/half-open），按依赖实例化。参数入 Settings：失败率阈值或连续失败数、开启时长、半开试探数。时间源可注入（构造函数接受 `clock: Callable[[], float]`），测试用假时钟，**禁止 sleep 型测试**

### 2b. 降级链路（写成表，逐条实现 + 逐条测试）

| 故障 | 系统行为 | 对外表现 |
|---|---|---|
| reranker 不可用/熔断 | 跳过重排，用融合结果直出 | HTTP 200；响应 meta `degraded: ["reranker"]`；日志 WARN |
| 向量库或 embedding 不可用 | 退化为纯 BM25 检索 | 200 + `degraded: ["vector"]` |
| LLM 不可用 | 返回模板化消息 + top-k 检索片段原文 | 200 + `degraded: ["llm"]` |
| 全部不可用 | 明确失败 | 503 + 结构化错误体（含 trace 提示） |

- SSE 场景下 `degraded` 信息放入起始或结束事件的 meta（具体位置沿用契约测试固化的事件结构，若需扩字段属于"契约变更"，在执行计划中显式声明）

### 2c. 入口限流

- 进程内令牌桶中间件：容量与速率入 Settings；超限返回 429 + `Retry-After`
- ADR 中注明：多实例部署时需迁移为 Redis 令牌桶（Phase 4 提供基础，但迁移不在本阶段范围）

### 2d. 故障注入开关（供测试与演示）

- Settings：`fault_inject: str = ""`（如 `"llm"`/`"reranker"`/`"vector"`），装配时用故障包装器包住对应客户端（抛出超时异常）。仅限非生产用途，README 注明

## 3. 范围外

- 不引 Redis（Phase 4）、不做分布式限流、不接 Sentinel/Resilience4j 类比的第三方全家桶
- 不做请求排队/舱壁隔离（BACKLOG 记录，量级不需要）

## 4. 实施细节规范

- 重试与熔断作为 infra 客户端的包装层（装饰器或组合），**业务 services 感知不到重试的存在**，只感知"该依赖当前不可用"的异常类型
- 定义统一异常：`DependencyUnavailable(dep_name)`——services 的降级逻辑只针对它分派
- 重试次数、熔断状态变化必须打日志（Phase 5 会结构化，本阶段普通 logging 即可，但字段要想好：dep、attempt、state_from、state_to）

## 5. 验收标准

| # | 标准 | 命令 | 期望 |
|---|---|---|---|
| C1 | 重试行为正确 | `pytest tests/unit/test_resilience.py -q` | 含用例：超时重试 2 次后抛出；400 不重试；429 重试；退避间隔用假时钟断言 |
| C2 | 熔断三态正确 | 同上 | 连续失败→open；open 期间调用被短路（fake 调用计数不增）；冷却后 half-open→成功则 closed |
| C3 | 四条降级链路 | `pytest tests/contract/test_degradation.py -q` | 表 2b 每行一个用例，断言状态码 + degraded 字段 + fake 调用路径 |
| C4 | 手动演示可复现 | `FAULT_INJECT=llm bash scripts/smoke.sh` 变体 | 返回 200，响应含降级标记与检索片段 |
| C5 | 限流生效 | 循环 curl 超过速率 | 出现 429 且带 Retry-After；速率内恢复 200 |
| C6 | 无裸调用残留 | `grep -rn "requests.post\|\.call(" app/infra/ \| grep -v resilience`（按实际 SDK 调用形态调整） | 所有出站调用均经防护层（报告中人工核对外部调用点清单逐条打勾） |
| C7 | 不变量 | `pytest -q` + 冒烟 + 评测对比 | 全部达标（happy path 行为不变） |

## 6. 测试与调试方法

- 假时钟模式：`FakeClock` 从 0 开始，`advance(seconds)` 推进；熔断测试全靠它，一秒都不许真睡
- 调试熔断状态：临时在日志中打印状态机转换（本就要求的日志字段），`grep "state_from" ` 追踪
- 手动模拟真实故障（不靠注入开关）：把 Settings 里对应依赖的 base_url 改成不可达地址，观察超时→重试→熔断全过程日志

## 7. 交付物清单

- [ ] PR + 阶段报告（含降级演示的 curl 原样输出）
- [ ] `docs/decisions/0003-resilience.md`：为什么手写熔断器、重试只对读操作的幂等性论证、参数取值理由
- [ ] REFACTOR_PLAN 状态表更新
