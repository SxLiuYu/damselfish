# 流式并行回退设计

## 概述

为千机 (Damselfish) 添加真正的流式支持，并在 Phase 2 并行竞速时支持流式首块竞速。

## 决策记录

| 问题 | 决策 |
|------|------|
| 首块判定标准 | 收到第一个 `data: {...}` SSE 行即胜出 |
| Phase 1 失败时机 | 只在首块前判定 429/504 触发回退；首块后不再回退 |
| 流式记忆保存 | 内存累积所有 chunk，流结束后保存完整 message |
| SSE 格式 | 标准化为 OpenAI 兼容格式，统一字段 |
| Phase 1 流式 | 所有流式请求均为真流式（`stream: true`） |

## 架构

```
客户端请求 (stream: true)
    │
    ▼
┌───────────────────────────────────────┐
│  app.py: chat_completions             │
│  wants_stream → 走流式路径             │
│  StreamingResponse(stream_chunks())   │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  router.stream_complete()             │
│  AsyncIterator[dict] + final_result   │
│                                       │
│  Phase 1: 最优目标 stream: true        │
│    ├─ 收到首块 → yield 首块，再转发后续 │
│    └─ 首块前 429/504 → Phase 2        │
│                                       │
│  Phase 2: _race_stream() 并行竞速首块   │
│    ├─ N 个候选同时 stream: true         │
│    ├─ 第一个收到有效 chunk 的胜出        │
│    └─ 取消其余，转发胜出者后续块          │
│                                       │
│  Phase 3: _serial_fallback()           │
│  (复用现有非流式回退，转单块 SSE)         │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  app.py: 流结束后累积完整 body          │
│  → 保存会话记忆                        │
│  → 返回 StreamingResponse 给客户端      │
└───────────────────────────────────────┘
```

## 新增 API

### `router.py` — 新增方法

#### `stream_complete(payload, context, session_id) -> AsyncIterator[dict]`

流式版本的 `complete()`。返回异步迭代器，每次 yield 一个标准化 chunk dict。流结束后通过 `StopIteration` 携带 `CompletionResult`（或通过参数回调）。

**算法：**

1. `rank_targets()` 获取排序后目标列表
2. **Phase 1**: 调用 `_stream_call(primary, payload)`
   - 尝试读取第一个 SSE chunk
   - 如果成功：yield 所有 chunk（包括首块），返回 `CompletionResult`
   - 如果首块是 429/504 且 targets 有候选：记录失败，进入 Phase 2
   - 如果首块是其他错误：抛 `NoTargetAvailable`
3. **Phase 2**: 调用 `_race_stream(candidates, payload)`
   - 并行发起 N 个 `_stream_call`
   - 第一个 yield 有效 chunk 的胜出
   - 取消其余
   - yield 胜出者的所有 chunk
   - 返回 `CompletionResult`
   - 全失败：返回 None
4. **Phase 3**: 如果 Phase 2 返回 None，走 `_serial_fallback()`（非流式，单块 SSE 转流式）

#### `_stream_call(target, payload) -> AsyncIterator[dict]`

发送 `stream: true` 请求，逐行解析 SSE 响应。

1. 构建请求 payload，设置 `stream: true`
2. 添加认证头
3. `await self.client.post(url, headers=headers, json=request)` 获取响应
4. 检查 HTTP 状态码
   - 非 2xx：抛 `UpstreamFailure`
5. 逐行读取 `response.aiter_lines()`
   - 解析 `data: {...}` 行
   - 标准化 chunk 字段（`id`, `object`, `created`, `model`, `choices`）
   - 替换 `model` 为 `target.model`
   - yield 每个 chunk
   - 遇到 `data: [DONE]` 停止
6. 异常处理：`httpx.TimeoutException` → 504, `httpx.HTTPError` → 502

**关键设计：** `_stream_call` 内部使用 `response.aiter_lines()` 读取上游 SSE，每个 chunk 独立 yield。首块读取发生在 `_stream_call` 内部，但 Phase 1 的"首块检查"在 `stream_complete` 中完成。

为支持"首块前回退"，`_stream_call` 需要区分"首块前"和"首块后"：
- 首块前：`_stream_call` 在第一个 `async for` 循环之前抛异常
- 首块后：`_stream_call` 已 yield 至少一个 chunk，后续异常不再触发回退

**实现方式：** `_stream_call` 内部使用一个 `asyncio.Event` 标记"首块已发送"，在 yield 第一个 chunk 前设置该事件。如果 `stream_complete` 收到异常时此事件未设置，则判定为"首块前"。

#### `_race_stream(candidates, payload) -> AsyncIterator[dict] | None`

并行竞速首块。

1. 为每个候选创建 `_stream_call` 的 async generator
2. 用 `asyncio.create_task(_first_stream_chunk(agen))` 包装每个 generator
3. 用 `asyncio.wait(FIRST_COMPLETED)` 等待第一个成功获取首块的任务
4. 胜出后：返回该 generator 的剩余部分（通过 `__anext__` 继续 yield）
5. 取消其余任务和 generator
6. 超时 / 全失败 → 返回 None

**关键设计：** `_first_stream_chunk` 读取 generator 的第一个 chunk 并缓存，然后返回 `(generator, first_chunk)`。这样首块不会丢失。

### `app.py` — 流式路径

```python
if wants_stream:
    async def stream_chunks():
        final_result = None
        complete_accumulator = []
        async for chunk in router.stream_complete(payload, context, decision_session):
            complete_accumulator.append(chunk)
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        # 流结束后，累积完整 body 保存记忆
        final_result = ...  # 从 stream_complete 获取
        if memory_enabled:
            _save_memory(final_result, ...)
    
    return StreamingResponse(stream_chunks(), ...)
```

**记忆保存时机变更：** 当前代码在 `complete()` 返回后保存记忆。流式模式下，记忆保存移到流结束后，在 `stream_chunks()` generator 的末尾。

### SSE 标准化函数

```python
def _normalize_chunk(chunk: dict, target_model: str) -> dict:
    """标准化上游 SSE chunk 为 OpenAI 格式。"""
    model = chunk.get("model", target_model)
    # 确保 choices[0].delta 存在
    choices = chunk.get("choices", [])
    if not choices:
        return chunk
    # 标准化字段
    return {
        "id": chunk.get("id", uuid.uuid4().hex),
        "object": "chat.completion.chunk",
        "created": chunk.get("created", int(time.time())),
        "model": model,
        "choices": [
            {
                "index": c.get("index", 0),
                "delta": c.get("delta", c.get("message", {})),
                "finish_reason": c.get("finish_reason"),
            }
            for c in choices
        ],
    }
```

## 模块边界

| 模块 | 职责 | 不包含 |
|------|------|--------|
| `router.py` | 流式请求、并行竞速、首块胜出决策 | SSE 序列化、HTTP 响应头 |
| `app.py` | SSE 序列化、StreamingResponse 构建、记忆保存 | 上游请求、竞速逻辑 |

## 错误处理

| 场景 | 行为 | HTTP 状态码 |
|------|------|------------|
| Phase 1 首块前 429/504 | 进入 Phase 2 并行竞速 | 200 (由胜出者决定) |
| Phase 1 首块前其他错误 | 报错 | 503 |
| Phase 1 首块后错误 | 连接中断，客户端收到不完整流 | 200 (流中断) |
| Phase 2 全部失败 | 进入 Phase 3 串行回退 | 200 (由胜出者决定) |
| Phase 2 超时 | 进入 Phase 3 串行回退 | 200 (由胜出者决定) |
| 全部失败 | 503 | 503 |

## 测试

| 测试 | 描述 |
|------|------|
| `test_stream_phase1_success` | Phase 1 流式正常返回，客户端收到逐块 SSE |
| `test_stream_phase1_429_fallback` | Phase 1 429，Phase 2 竞速胜出，客户端收到胜出者流 |
| `test_stream_phase1_504_fallback` | Phase 1 超时，Phase 2 竞速胜出 |
| `test_stream_phase2_race_first_chunk` | 多个候选，第一个发首块的胜出 |
| `test_stream_phase2_all_fail` | 所有并行候选失败，走 Phase 3 串行回退 |
| `test_stream_normalize_chunk` | 上游非标准格式被标准化为 OpenAI 格式 |
| `test_stream_memory_save` | 流结束后完整 assistant message 被保存到 SQLite |

## 对现有代码的影响

1. **`router.py`**: 新增 `stream_complete()`、`_stream_call()`、`_race_stream()`，约 150 行。现有 `complete()` 和 `_race_targets()` 不变。
2. **`app.py`**: 修改 `chat_completions` 的流式路径，新增记忆保存逻辑。约 30 行变更。
3. **`tests/test_router.py`**: 新增 7 个测试，约 200 行。
4. **`config.py`**: 无需修改（复用现有并行回退参数）。

## 约束

- 流式请求不压缩对话（`_compress_conversation` 只在非流式路径触发）
- 流式请求的 `X-Damselfish-Latency-Ms` 为首块到达时间
- 不支持 `usage` 字段在流式响应中（上游可能只在最后 chunk 带 usage）