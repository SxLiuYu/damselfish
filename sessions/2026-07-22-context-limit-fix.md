# 会话记录：千机 (Damselfish) 上下文超限修复

**日期**: 2026-07-22 14:33:47  
**项目**: 千机 (Damselfish) — 本地优先、OpenAI API 兼容的智能模型路由器  
**部署**: 北京服务器 (deploy/beijing/)  
**工作目录**: `C:/Users/mi/orca/workspaces/免费的才是最好的/main-2`

---

## 1. 错误信息

```
Error: primary target zhipu-glm-4v-flash failed: HTTP 400 API 调用参数有误，请检查文档。
Input validation error: `inputs` tokens + `max_new_tokens` must be <= 16384.
Given: 21790 `inputs` tokens and 1024 `max_new_tokens`
```

## 2. 根因分析

千机的路由逻辑 (`selector.py` 的 `rank_targets()`) 只检查能力匹配和场景匹配，**没有考虑各个模型的上下文长度限制**。当同一会话积累了大量历史消息后（或 `project_memory` 注入了跨会话上下文），输入 token 数可能超过某些模型的上下文窗口（如智谱 `glm-4v-flash` 上限 16384），导致请求直接被上游拒绝，返回 HTTP 400。

当前 `TargetConfig` 中没有 `max_context` 字段，路由器无法提前排除承载不了的长上下文目标。

## 3. 修复方案

### 3.1 方案 A：节点声明上下文上限 + 路由时过滤（核心修复）

#### 修改文件

| 文件 | 变更 |
|------|------|
| `damselfish/config.py` | `TargetConfig` 新增 `max_context: int \| None = None`；`target_from_mapping()` 解析该字段 |
| `damselfish/selector.py` | `RouteContext` 新增 `estimated_input_tokens: int = 0`；`infer_context()` 调用 `_estimate_messages_tokens()` 估算输入 token；`rank_targets()` 新增 `max_new_tokens` 参数，过滤 `estimated_input_tokens + max_new_tokens > target.max_context` 的目标 |
| `damselfish/router.py` | `_upstream_payload()` 增加安全封顶逻辑：当 `target.max_context` 存在时，自动下调 `max_tokens`/`max_completion_tokens` 使总和不超过上限；`complete()` 和 `stream_complete()` 调用 `rank_targets()` 时传入 `max_new_tokens`；新增 `_is_context_overflow()` 检测 400 超长错误 |
| `damselfish/nodes.py` | `normalize_node()` 支持 `max_context` 字段；`public_node()` 返回 `max_context`；新增 `_optional_int()` 辅助函数 |
| `scripts/sync_free_models.py` | `common_node()` 新增 `max_context` 参数；新增 `ZHIPU_CONTEXT_LIMITS` 字典记录已知智谱免费模型的上下文限制；`discover_zhipu()` 自动为 `glm-4v-flash` 等模型设置 `max_context=16384` |

### 3.2 方案 B：400 超长错误可回退（兜底）

在 `router.py` 中，`complete()` 和 `stream_complete()` 的回退条件从 `error.status not in (429, 504)` 扩展为 `error.status not in (429, 504) and not _is_context_overflow(error)`。当检测到 400 且错误信息包含 "max_new_tokens"、"must be <="、"context length" 等关键词时，允许回退到下一个有足够上下文的目标。

## 4. 技术细节

### Token 估算算法

```python
def _estimate_messages_tokens(messages):
    """
    粗略估算：CJK 字符约 1.5 token/字，ASCII 约 0.25 token/字
    混合估算：tokens ≈ len(content) / 2.5
    每条消息额外 +4 token 结构开销（role 标记等）
    """
```

### 智谱已知上下文限制

```python
ZHIPU_CONTEXT_LIMITS = {
    "glm-4v-flash": 16384,
    "glm-4v": 16384,
}
```

### 安全封顶逻辑

```python
# 在 _upstream_payload() 中
if target.max_context is not None:
    inputs_tokens = _estimate_current_input_tokens(messages)
    max_new = request.get("max_tokens", 1024)
    allowed = target.max_context - inputs_tokens
    if max_new > allowed:
        request["max_tokens"] = max(1, int(allowed))
```

## 5. 项目架构概览

### 核心模块

| 模块 | 职责 |
|------|------|
| `damselfish/app.py` | FastAPI 应用，HTTP 路由，记忆管理，流式 SSE 处理 |
| `damselfish/router.py` | 模型路由器，三级回退（串行→并行竞速→串行回退），流式支持 |
| `damselfish/selector.py` | 场景/人物推断，目标排序，token 估算 |
| `damselfish/config.py` | 配置加载，TargetConfig/RouteRule/PersonaRule 数据类 |
| `damselfish/store.py` | SQLite 持久化，指标统计，会话记忆，项目上下文 |
| `damselfish/nodes.py` | 节点管理，CRUD API，节点测试，模型发现 |
| `damselfish/git_sync.py` | Git 记忆同步 |
| `damselfish/cli.py` | CLI 入口 |

### 部署配置 (deploy/beijing/)

- **config.yml**: 北京服务器配置，端口 18086，上游 LLM Router 在 8200
- **damselfish.service**: systemd 服务单元
- **nginx.conf**: Nginx 反向代理配置
- **nginx-location.conf**: Nginx location 块

### 模型池

- 本地 Qwen3 8B (http://127.0.0.1:8080/v1)
- Agnes 2.5 Flash / 2.0 Flash (agnes-ai.com)
- GLM 4.7 Flash (open.bigmodel.cn)
- DeepSeek V4 Pro (finna.com.cn)
- 自动同步的免费模型 (Kilo, Pollinations, Zhipu)

### 路由策略

1. **Phase 1**: 最优目标串行尝试
2. **Phase 2**: 429/504/400(超长) → 并行竞速（最多 3 个候选，30s 超时）
3. **Phase 3**: 全部失败 → 串行回退剩余目标

### 记忆系统

- SQLite 持久化会话记忆
- 项目级跨会话共享上下文
- Git 同步多设备记忆
- 长对话自动压缩（>30 条消息时）

## 6. 修改的文件清单

1. `damselfish/config.py` — TargetConfig 新增 max_context
2. `damselfish/selector.py` — RouteContext 新增 estimated_input_tokens，rank_targets 过滤，_estimate_messages_tokens
3. `damselfish/router.py` — _upstream_payload 安全封顶，_is_context_overflow 检测，_max_new_tokens，_estimate_current_input_tokens
4. `damselfish/nodes.py` — normalize_node 支持 max_context，public_node 返回 max_context，_optional_int
5. `scripts/sync_free_models.py` — common_node 支持 max_context，ZHIPU_CONTEXT_LIMITS，discover_zhipu 自动设置

## 8. 测试结果

全部 32 个测试通过，无破坏性变更：

```
tests/test_app.py (3 passed)
tests/test_git_sync.py (3 passed)
tests/test_nodes.py (9 passed)
tests/test_router.py (10 passed)
tests/test_selector.py (2 passed)
tests/test_store.py (3 passed)
======================= 32 passed, 1 warning in 15.88s =======================
```

### 第二轮：CJK 感知估算修正 + 36 个新测试

- **问题**: 原估算 `len/2.5` 对中文过于乐观，导致 70655 tokens 仍未被过滤
- **修复**: 改用 CJK 字符感知估算，识别中文字符按 1.5 tokens/char，非 CJK 按 0.25 tokens/char，附加 10% 安全裕量
- **新增测试**: 36 个新测试，覆盖 token 估算、max_context 过滤、安全封顶、400 回退、节点管理、端到端

```
tests/test_app.py (4 passed)
tests/test_git_sync.py (3 passed)
tests/test_nodes.py (13 passed)
tests/test_router.py (32 passed)
tests/test_selector.py (13 passed)
tests/test_store.py (3 passed)
======================= 68 passed, 1 warning in 4.17s =======================
```

### 北京服务器部署状态

- 代码已推送至 GitHub 并拉取到服务器
- `managed-nodes.json` 中 6 个智谱节点已设置 `max_context`
- 服务已重启，健康检查通过，21 个目标全部可用
- 已生效的 `max_context` 配置：

| 节点 | 模型 | max_context |
|------|------|------------|
| zhipu-glm-4v-flash | glm-4v-flash | 16384 |
| zhipu-glm-4.1v-thinking-flash | glm-4.1v-thinking-flash | 16384 |
| zhipu-glm-4.6v-flash | glm-4.6v-flash | 16384 |
| zhipu-glm-4-flash-250414 | glm-4-flash-250414 | 128000 |
| zhipu-glm-4.5-flash | glm-4.5-flash | 128000 |
| zhipu-glm-4.7-flash | glm-4.7-flash | 128000 |

## 9. 部署步骤

1. ✅ 将修改后的代码推送到北京服务器 (`git push origin main`)
2. ✅ 服务器拉取代码 (`cd /opt/damselfish && git pull`)
3. ✅ 运行测试确认无破坏 (`68 passed`)
4. ✅ 更新 `managed-nodes.json` 中智谱节点的 `max_context`
5. ✅ 重启服务：`systemctl restart damselfish.service`
6. ✅ 验证：`curl http://127.0.0.1:18086/health` 返回 `status: ok`

## 10. 后续建议

- 为其他免费模型（Kilo、Pollinations）也添加 `max_context` 限制
- 考虑在管理页面展示每个节点的 `max_context` 信息
- 监控日志中的 "capping max_new_tokens" 警告，调整估算算法精度
- 为智谱其他模型（如 glm-4-plus、glm-4-flash 等）补充上下文限制
