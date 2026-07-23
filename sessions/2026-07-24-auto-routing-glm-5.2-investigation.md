# 千机 `auto` 路由延迟与 GLM-5.2 模型改写排查

**日期**: 2026-07-24

**项目**: 千机 (Damselfish)

**范围**: 北京部署、Damselfish 到本地 LLM Router 的请求链

**状态**: 根因已定位，尚未修改生产配置或代码

## 摘要

用户观察到直连 `glm-5.2` 响应很快，但通过千机使用 `auto` 时首包可能需要 90 秒以上。排查确认这不是 GLM-5.2 本身变慢，而是请求经过了两层自动路由，并可能在命中 GLM-5.2 前等待其他免费节点失败。

更关键的是，即使客户端显式发送 `model: glm-5.2`，当前千机实现也会在构造上游请求时将模型无条件改写为北京 target 配置的 `auto`。因此，“客户端默认选择 GLM-5.2”不等于“本地 LLM Router 实际收到 GLM-5.2”。

## 请求链

当前北京部署只有一个 Damselfish target：

```text
客户端
  -> Damselfish
  -> beijing-llm-router
  -> http://127.0.0.1:8200/v1
  -> model: auto
  -> 免费/公共模型池动态选择
```

北京配置 `deploy/beijing/config.yml`：

```yaml
targets:
  - id: beijing-llm-router
    base_url: http://127.0.0.1:8200/v1
    model: auto
```

该配置从北京部署文件首次提交时就是 `model: auto`，Git 历史中没有将其配置为 `glm-5.2` 的记录。

## 根因

### 1. 显式模型被 target 模型覆盖

`damselfish/router.py` 中 `_upstream_payload()` 无条件执行：

```python
request["model"] = target.model
```

因此实际行为是：

```text
客户端请求：model = glm-5.2
        ↓
Damselfish 选择唯一 target：beijing-llm-router
        ↓
构造上游请求：model = target.model
        ↓
target.model = auto
        ↓
本地 LLM Router 收到：model = auto
```

`rank_targets()` 虽然会读取客户端请求的模型，但它只使用该字段给匹配的 Damselfish target 降低排序分数。北京配置中没有独立的 `glm-5.2` target，只有模型为 `auto` 的 `beijing-llm-router`，所以显式指定 `glm-5.2` 无法命中具体 target，也无法透传给下游。

### 2. 二级 `auto` 会先尝试其他节点

一次已观测慢请求的过程为：

```text
kilo-kwaipilot-kat-coder-pro-v2-5
  -> 等待约 40 秒
  -> HTTP 429
  -> 进入并行 fallback
  -> Finna 的 glm-5.2 成功
```

该请求最终成功节点的模型调用延迟约为 23.7 秒，但客户端首包约为 90.8 秒。差值主要来自前序失败、fallback 调度、排队和服务器响应延迟。

另有 `zhipu-glm-4.5-flash` 成功请求耗时约 71.6、99.7 和 129.9 秒，说明 `auto` 并不保证优先选择 GLM-5.2。

### 3. 响应头没有展示端到端耗时

`X-Damselfish-Latency-Ms` 记录的是最终获胜 `_call` 的耗时，其中可能包含该 target 的 semaphore 排队，但不包含此前失败 target 的等待时间，也不代表客户端端到端首包时间。

因此响应最终显示 GLM-5.2，只能证明它是成功返回的节点，不能证明请求从一开始就直接调用了 GLM-5.2。

## 现场数据

排查时，本地 `beijing-llm-router` target 的统计为：

```text
requests: 5869
successes: 5580
failures: 289
consecutive_failures: 19
ewma_latency_ms: 44779
last_error: timeout
circuit_open: true
```

多类免费节点存在重复超时或 HTTP 429，包括 Kilo Auto、OpenRouter Free、Nemotron、Poolside、GLM 4.7，以及历史上的 Finna 节点。

北京服务还存在独立的应用或主机调度延迟：

```text
公开接口：
/health      约 11.2 秒
/stats       约 8.4 秒
/v1/models   约 4.3 秒

通过既有 SSH 隧道直连 127.0.0.1:18086：
/health      约 5.8 秒
/stats       约 6.5 秒
简单 401     约 4.5 秒
```

TCP 建连较快，但应用响应较慢。该问题与上游模型延迟并存，只修模型路由不能消除全部延迟。

## 为什么直连 GLM-5.2 快

用户现场确认直连 `glm-5.2` 响应很快。直连绕过了以下步骤：

- Damselfish 将模型改写为 `auto`
- 二级路由的免费节点选择
- 高失败节点的串行等待
- 429/504 后的并行 fallback
- fallback 失败后的剩余节点串行尝试

所以直连快与 `auto` 慢并不矛盾，反而进一步说明主要延迟发生在路由和失败重试阶段，而不是 GLM-5.2 推理阶段。

## 建议修复

### 短期止血：固定默认模型

如果北京部署的默认模型确实应该是 GLM-5.2，可将 target 配置改为：

```yaml
model: glm-5.2
```

修改前需先通过 `http://127.0.0.1:8200/v1/models` 确认本地 LLM Router 接受的准确模型 ID。修改后使用同一组请求对比直连与千机的首包时间、总耗时和实际模型。

该方案会让所有经过此 target 的请求固定使用 GLM-5.2，不再执行二级 `auto` 选择。

### 长期修复：区分默认模型与显式模型

建议调整模型语义：

```text
请求 model=damselfish/auto
  -> 使用 target 配置的默认模型或执行 Damselfish 路由

请求 model=glm-5.2
  -> 仅选择支持该模型的 target
  -> 对允许模型透传的 target，将 glm-5.2 原样发送给上游
```

实现时需要避免对所有上游盲目透传模型名，因为不同供应商可能使用不同模型 ID。更稳妥的方式是在 target 配置中明确声明可接受模型、默认模型以及是否允许透传。

建议同时补充：

1. 为显式模型透传添加单元测试和端到端测试。
2. 在 `/v1/models` 中暴露可实际请求的具体模型，而不只是 `damselfish/auto` 和 target ID。
3. 记录原始请求模型、发往上游的模型、每次尝试耗时和端到端耗时。
4. 加强近期失败率、连续失败和 429 冷却惩罚。
5. 为整个请求设置共享 deadline，避免多轮超时累加到 90 至 180 秒。
6. 将真正的流式请求改为 `client.stream(...)`，避免在首块返回前缓冲完整响应。

## 后续验证清单

- 确认本地 LLM Router 的 GLM-5.2 精确模型 ID。
- 分别发送 `model: auto` 和 `model: glm-5.2`，记录 LLM Router 实际收到的请求体。
- 对同一提示执行至少 20 次直连和千机对比，统计成功率、P50/P95 首包和总耗时。
- 检查北京主机 CPU、内存、交换、磁盘 IO、连接和 Uvicorn 调度情况。
- 检查 `damselfish` 与 `llm-router` 服务日志，确认超时发生在哪一层。
- 确认没有 Git 进程后，再处理 `.git/config.lock` 遗留锁问题。

## 结论

当前北京部署并没有在 Damselfish 层固定使用 GLM-5.2。客户端显式选择 GLM-5.2 时，请求模型仍会被 target 配置覆盖为 `auto`，随后由本地 LLM Router 在免费节点池中重新选择。慢请求经常先等待其他节点超时或限流，最后才由 GLM-5.2 返回。

因此本次问题的直接根因是模型改写与二级 `auto` 路由，而非 GLM-5.2 本身。北京服务器的应用响应延迟是另一个需要并行排查的问题。
