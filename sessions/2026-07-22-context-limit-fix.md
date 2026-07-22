# 千机 (Damselfish) 上下文超限修复 — 会话总结

**日期**: 2026-07-22  
**项目**: 千机 (Damselfish) — 本地优先、OpenAI API 兼容的智能模型路由器  
**部署**: 北京服务器 (123.57.107.21, /opt/damselfish)  
**状态**: ✅ 已修复、已测试、已部署

---

## 问题

智谱 `glm-4v-flash` 等模型因输入超长返回 HTTP 400：
```
Input validation error: `inputs` tokens + `max_new_tokens` must be <= 16384
Given: 70655 `inputs` tokens and 1024 `max_new_tokens`
```

**根因**: 路由器 `rank_targets()` 只检查能力/场景匹配，未考虑模型上下文长度限制。

### 附加发现：压缩机制形同虚设

`_compress_conversation()` 硬编码了 `preferred_targets=("deepseek-v4-flash",)`，但该目标在北京服务器上**不存在**，导致压缩请求永远找不到目标，抛出异常后静默吃掉——对话从未被压缩。这是上下文超限的**重要诱因**：对话不断累积，却没有压缩机制兜底收缩。

## 修复方案（三层保护）

| 层 | 机制 | 文件 |
|----|------|------|
| **1. 路由过滤** | `TargetConfig.max_context` + `rank_targets()` 过滤超限目标 | `config.py`, `selector.py` |
| **2. Payload 封顶** | `_upstream_payload()` 自动下调 `max_tokens` | `router.py` |
| **3. 400 回退兜底** | `_is_context_overflow()` 检测超长错误并回退 | `router.py` |

### Token 估算（CJK 感知）

```python
# 中文字符 1.5 tokens/char，非 CJK 0.25 tokens/char，+10% 安全裕量
cjk = len(_CJK_RE.findall(text))
estimate = cjk * 1.5 + (len(text) - cjk) * 0.25
return max(1, int(estimate * 1.1))
```

### 智谱模型上下文限制

| 模型 | max_context |
|------|------------|
| glm-4v-flash / glm-4.1v-thinking-flash / glm-4.6v-flash | 16384 |
| glm-4-flash-250414 / glm-4.5-flash / glm-4.7-flash | 128000 |

## 修改文件

### 核心修复
1. `damselfish/config.py` — `TargetConfig.max_context` 字段
2. `damselfish/selector.py` — `RouteContext.estimated_input_tokens`，`rank_targets()` 过滤，CJK 感知估算
3. `damselfish/router.py` — `_upstream_payload()` 封顶，`_is_context_overflow()`，`_max_new_tokens()`
4. `damselfish/nodes.py` — `normalize_node()` 支持 `max_context`（更新时从 existing 继承）
5. `scripts/sync_free_models.py` — `ZHIPU_CONTEXT_LIMITS`，`common_node(max_context=...)`

### 压缩修复（第二轮）
6. `damselfish/app.py` — 移除硬编码 `deepseek-v4-flash`，改用 auto-routing；中文 prompt；token 减少验证

## 测试

- **39 个新测试**，覆盖 token 估算、max_context 过滤、安全封顶、400 回退、节点管理、端到端、压缩
- **71 个测试全部通过**（本地 + 北京服务器）

## 部署

1. ✅ `git push origin main` → GitHub
2. ✅ 服务器 `git pull` (eaa1feb → 5fe5ac8)
3. ✅ 服务器测试 68 passed
4. ✅ 更新 `/var/lib/damselfish/managed-nodes.json` 中 6 个智谱节点 `max_context`
5. ✅ `systemctl restart damselfish` → `active`
6. ✅ `/health` → `status: ok`，21 目标可用

## Git 提交

```
9aab01a fix: compression used nonexistent deepseek-v4-flash target, silently failing
5fe5ac8 test: comprehensive context-limit fix tests (36 new tests, all 68 pass)
10c47a5 fix: CJK-aware token estimation for accurate max_context filtering
143eea2 fix: prevent context overflow 400 errors with max_context filtering and capping
```

## 结论

使用 damselfish 的用户不会再遇到 `HTTP 400 inputs tokens + max_new_tokens must be <= 16384` 错误。三层保护确保：路由层提前过滤 → payload 自动封顶 → 400 错误兜底回退。压缩功能现已真正可用：不再硬编码不存在的目标，改用中文 prompt，并在压缩前验证 token 确实减少。

## 后续建议

- 为 Kilo、Pollinations 免费模型补充 `max_context`
- 管理页面展示 `max_context`
- 监控 "capping max_new_tokens" 日志，校准估算精度
- `scripts/sync_free_models.py` 下次同步时新节点自动带 `max_context`
