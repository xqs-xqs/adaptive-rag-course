# Phase 2 — 测试 + CI 门禁

- 分支：`refactor/phase-2-tests-ci`
- 前置：Phase 1 已合并（接口抽象已存在，这是本阶段能用 fake 的前提）
- 本阶段性质：建立防回归的安全网。**从本阶段起，后续所有阶段的改动都受测试保护。**

## 0. 前置假设核对

| # | 假设 | 验证命令 |
|---|---|---|
| A1 | Phase 1 目录结构就位 | `ls app/infra/ app/services/ app/domain/` |
| A2 | 四个抽象基类存在 | `grep -rn "class.*ABC" app/infra/` |
| A3 | 仓库为 GitHub 公开仓库，Actions 可用 | 人类确认 |
| A4 | 评测入口可用 | `python scripts/run_eval.py --help` |

## 1. 涉及的现状文件

- 新增：`tests/`（结构见下）、`.github/workflows/ci.yml`、`pyproject.toml`（ruff/mypy/pytest 配置）
- 修改：`requirements.txt` 或新增 `requirements-dev.txt`（pytest、pytest-socket、ruff、mypy、httpx）
- 修改：`scripts/run_eval.py`（支持 `--subset` 小样本模式与 `--compare-baseline` 对比输出）

## 2. 目标行为

### 2a. 测试结构与必须覆盖的用例（枚举制，不设覆盖率虚荣指标）

```
tests/
  fakes/
    fake_llm.py            # 返回固定/可编程文本；记录调用次数与入参
    fake_embedding.py      # 基于 sha256 的确定性伪向量；记录调用次数
    fake_vectorstore.py    # 内存实现，支持按 id 增删查
  unit/
    test_fusion_rrf.py     # ≥3 用例：已知两路排名→已知融合结果；并列分处理；k 参数影响
    test_diversity.py      # 同源文档超限被截断；不同源不受影响（覆盖 Phase 0 修过的分支）
    test_chunking.py       # parent-child 关联完整性；上下文前缀注入存在且格式正确；边界（空文档/超长段）
    test_router.py         # 用 fake LLM 返回各意图标签 → 断言走到对应检索策略（以 fake 调用记录为证）
    test_lexical.py        # jieba 分词稳定性：固定输入→固定 token 序列；中英混合查询
    test_config.py         # 缺 key 时 Settings 抛错；默认值与 ARCHITECTURE 清单一致（抽 3 项）
  contract/
    test_api_contract.py   # TestClient + 全套 fake 注入：POST 问答接口 → 200；SSE 事件序列符合约定
                           # 把当前真实响应结构固化为契约（字段名、事件类型、结束标记）
```

- fake 优先于 mock（fake 是可复用的行为实现，mock 断言调用是补充手段）
- **单元/契约测试零真实网络**：`pytest-socket` 全局禁用 socket，需要本地回环的契约测试单独放行

### 2b. CI（.github/workflows/ci.yml，三个 job）

1. `lint`：`ruff check .` + `mypy app/`（mypy 起步宽松：`ignore_missing_imports = true`，只强制 app/ 下的显式标注错误）
2. `test`：`pytest -q`，无任何 secret 注入（天然保证没人偷用真实 API）
3. `eval`：`workflow_dispatch` 手动触发，使用仓库 secret `DASHSCOPE_API_KEY`，跑 `--subset` 评测并与 `docs/baseline/` 对比，检索指标回退超 I1 阈值则 job 失败

### 2c. 评测工具化

- `--subset`：固定抽取评测集的一个子集（写死用例 id 列表，保证可比），控制 CI 费用
- `--compare-baseline`：输出与基线的逐指标差值表，超阈值时非零退出——这个退出码就是门禁

## 3. 范围外

- 不为 infra 的真实实现写集成测试（真实 API 的验证由评测和冒烟承担）
- 不追覆盖率百分比；不测私有函数实现细节（测行为，不测实现）
- 不动业务代码——若发现"不可测"的代码需要小改（如缺一个注入点），先报告获批，改动限制在最小

## 4. 实施细节规范

- 契约测试是**先固化现状**：字段名以当前真实响应为准，哪怕命名不佳也不改（改契约是显式决策，不在本阶段）
- fake_embedding 的伪向量必须维度与真实一致（从 Settings 读维度）
- conftest.py 提供 `app_with_fakes` fixture：完成全套依赖注入的 TestClient，contract 测试共用
- pyproject.toml 中 pytest 配置 `addopts = "-p no:cacheprovider --disable-socket --allow-unix-socket"`（按 pytest-socket 实际用法调整）

## 5. 验收标准

| # | 标准 | 命令 | 期望 |
|---|---|---|---|
| C1 | 全绿且快 | `time pytest -q` | 全过；总时长 < 60s（待校准） |
| C2 | 断网仍绿 | 断开网络后 `pytest -q`（或依赖 pytest-socket 报错证明无网络调用） | 全过 |
| C3 | 必须用例齐全 | `ls tests/unit/ tests/contract/` 对照 2a 清单 | 无缺文件；每文件用例数达标 |
| C4 | 门禁真的会拦 | 临时把 RRF 的 k 改错 → `pytest -q` | test_fusion_rrf 失败（验证完还原，此验证过程写入报告） |
| C5 | lint 通过 | `ruff check . && mypy app/` | 0 错误 |
| C6 | CI 三 job 就位 | 推分支后查看 Actions | lint/test 自动跑且绿；eval 可手动触发 |
| C7 | 评测对比工具可用 | `python scripts/run_eval.py --subset --compare-baseline` | 输出差值表；未回退时退出码 0 |
| C8 | 不变量 I1 | 全量评测对比基线 | 达标 |

## 6. 测试与调试方法

- 跑单个测试：`pytest tests/unit/test_fusion_rrf.py -k tie -x -q`；进断点：`pytest --pdb`
- SSE 契约测试技巧：`httpx` 的 stream 接口逐事件断言；先打印真实事件流一次，人工确认后固化为期望值
- CI 与本地不一致：优先怀疑依赖版本（CI 用 lock 安装）与时区/locale（jieba 无关，但日期格式化相关测试要用固定值）

## 7. 交付物清单

- [ ] PR + 阶段报告（含 C4 的"门禁演习"记录）
- [ ] `docs/decisions/0002-testing-strategy.md`：为什么枚举用例而非覆盖率门禁、为什么 fake 优先、评测为什么拆 subset
- [ ] REFACTOR_PLAN 状态表更新
