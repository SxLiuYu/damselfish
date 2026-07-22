# 上下文超限修复测试方案

> **Goal:** 全面验证千机 (Damselfish) 的 `max_context` 保护机制，确保不会再出现因上下文超长导致 HTTP 400 的错误。

## 修复范围

| 变更 | 文件 |
|------|------|
| `TargetConfig.max_context` 字段 | `damselfish/config.py` |
| `rank_targets()` 过滤超限目标 | `damselfish/selector.py` |
| CJK 感知的 token 估算 `_estimate_messages_tokens()` | `damselfish/selector.py` |
| `_upstream_payload()` 安全封顶 `max_tokens` | `damselfish/router.py` |
| `_is_context_overflow()` 检测 400 超长错误 | `damselfish/router.py` |
| `_estimate_current_input_tokens()` 与 `_estimate_text_tokens()` | `damselfish/router.py` |
| 400 超长可回退（`complete()` + `stream_complete()`） | `damselfish/router.py` |
| 节点管理支持 `max_context` | `damselfish/nodes.py` |
| 自动同步脚本设置 `max_context` | `scripts/sync_free_models.py` |

## 测试文件结构

| 文件 | 新增测试 |
|------|---------|
| `tests/test_selector.py` | token 估算、max_context 过滤 |
| `tests/test_router.py` | 安全封顶、400 超长回退 |
| `tests/test_nodes.py` | max_context 在节点 CRUD 中的传递 |
| `tests/test_app.py` | 端到端：超长上下文自动回退 |

## 测试清单

### 1. Token 估算测试 (`tests/test_selector.py`)

- [ ] `test_estimate_messages_tokens_pure_chinese` — 纯中文文本 token 估算≥实际值
- [ ] `test_estimate_messages_tokens_pure_english` — 纯英文文本估算合理
- [ ] `test_estimate_messages_tokens_mixed` — 中英文混合估算
- [ ] `test_estimate_messages_tokens_multimodal` — 多模态消息（图片+文本）只估算文本部分
- [ ] `test_estimate_messages_tokens_empty` — 空消息和空内容

### 2. max_context 过滤测试 (`tests/test_selector.py`)

- [ ] `test_rank_targets_filters_by_max_context` — 短上下文目标被过滤
- [ ] `test_rank_targets_passes_within_context` — 足够上下文的目标保留
- [ ] `test_rank_targets_no_max_context` — 未设置 max_context 的目标不受影响
- [ ] `test_rank_targets_max_new_tokens_parameter` — 不同的 max_new_tokens 影响过滤

### 3. 安全封顶测试 (`tests/test_router.py`)

- [ ] `test_upstream_payload_caps_max_tokens` — max_tokens 被安全封顶
- [ ] `test_upstream_payload_no_cap_within_limit` — 在限制内不封顶
- [ ] `test_upstream_payload_no_max_context` — 未设置 max_context 不封顶
- [ ] `test_upstream_payload_probe_skips_capping` — probe 请求跳过封顶

### 4. 400 超长回退测试 (`tests/test_router.py`)

- [ ] `test_router_falls_back_on_context_overflow_400` — 400 超长错误回退到下一个目标
- [ ] `test_router_fails_on_other_400` — 其他 400 错误不触发回退
- [ ] `test_stream_complete_phase1_400_context_overflow_fallback` — 流式 400 超长回退
- [ ] `test_router_context_overflow_marker_detection` — 各个 overflow marker 的检测

### 5. 节点管理测试 (`tests/test_nodes.py`)

- [ ] `test_node_create_with_max_context` — 创建节点时设置 max_context
- [ ] `test_node_public_exposes_max_context` — 公共 API 返回 max_context
- [ ] `test_node_update_preserves_max_context` — 更新节点保留 max_context

### 6. 端到端测试 (`tests/test_app.py`)

- [ ] `test_auto_fallback_to_longer_context_on_overflow` — 长上下文自动回退到支持的目标

---

## 测试数据

### 目标配置

```python
TargetConfig(
    "short-context", "Short", "http://short/v1", "short",
    local=True, priority=1, max_context=4096,
)
TargetConfig(
    "long-context", "Long", "http://long/v1", "long",
    local=True, priority=2, max_context=128000,
)
```

### 超长消息

```python
# 纯中文 42800 字符：约 70620 tokens
long_messages = [
    {"role": "user", "content": "中" * 42800}
]
```

### 400 超长错误响应

```python
httpx.Response(400, json={
    "error": {"message": "Input validation error: `inputs` tokens + `max_new_tokens` must be <= 16384"}
})
```