# Damselfish

Damselfish 是一个本地优先、OpenAI API 兼容的智能模型路由器。它把本地模型和多个免费云 API 组成动态模型池，根据任务能力、人物设定、实时延迟和错误率自动选择，并在 429、超时或服务错误时切换到下一个模型。

## 已实现

- `POST /v1/chat/completions` 和 `GET /v1/models`
- 根据工具、编程、推理、创作、翻译、视觉场景匹配模型能力
- 根据 system prompt 或请求参数识别人物设定
- EWMA 延迟、历史失败率、优先级综合排序
- 429/超时/5xx 自动回退和指数冷却熔断
- 后台低频健康及延迟探测，可按模型关闭避免浪费额度
- SQLite 持久化指标、路由决策和跨模型会话记忆
- 项目化多会话记忆，同项目不同会话共享近期上下文
- Git 不可变快照同步，多设备自动拉取、提交和推送
- 非流式响应及 OpenAI SSE 兼容响应
- `/health` 和 `/stats` 可观察端点
- 带 Bearer Key 验证的云节点添加、测试和热更新管理页面

## 快速启动

```bash
cd /Users/sxliuyu/repos/damselfish
cp config.example.yml config.yml
uv sync --extra test
export AGNES_API_KEY='替换为新密钥'
export ZHIPU_API_KEY='替换为新密钥'
export FINNA_API_KEY='替换为新密钥'
uv run damselfish --config config.yml
```

未设置密钥的云目标会自动退出候选池。本地 Qwen 默认地址是 `http://127.0.0.1:8080/v1`，模型 ID 应与该服务 `/v1/models` 返回值一致。不要把密钥写入 `config.yml` 或提交到 Git。

## 调用

```bash
curl -i http://127.0.0.1:8086/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Damselfish-Project: basketball-live' \
  -H 'X-Damselfish-Session: deploy-2026-07-18' \
  -d '{
    "model": "damselfish/auto",
    "messages": [{"role":"user","content":"帮我分析并修复这个 Python 错误"}]
  }'
```

响应头会显示实际选择：

- `X-Damselfish-Target`
- `X-Damselfish-Model`
- `X-Damselfish-Latency-Ms`
- `X-Damselfish-Scenario`
- `X-Damselfish-Project`
- `X-Damselfish-Session`
- `X-Damselfish-Memory-Sync`

同一个项目和会话的历史会存入 SQLite。即使请求从 Qwen 切换到 Agnes 或 GLM，新模型仍会收到同一份上下文。同项目其他会话的近期内容会作为共享项目记忆注入，但不会重复保存到当前会话。不传 session 时不保存服务端记忆。

也可以在 JSON 中显式指定：

```json
{
  "model": "damselfish/auto",
  "damselfish": {
    "project_id": "basketball-live",
    "project_title": "篮球比赛数据直播系统",
    "session_id": "cloud-deploy",
    "session_title": "云端部署",
    "persona": "developer",
    "scenario": "coding",
    "memory": true,
    "project_memory": true
  },
  "messages": [{"role": "user", "content": "检查部署状态"}]
}
```

## 多端记忆同步

推荐创建一个单独的 **GitHub 私有仓库** 保存记忆，不要和 Damselfish 源码仓库混用。会话可能包含代码、服务器信息和其他敏感数据，禁止使用公开仓库。

在每台设备使用相同的记忆仓库，并设置不同设备 ID：

```bash
export DAMSELFISH_MEMORY_GIT_URL='git@github.com:你的账号/damselfish-memory.git'
export DAMSELFISH_DEVICE_ID='mac-mini'
```

然后在 `config.yml` 开启：

```yaml
git_sync:
  enabled: true
  repository: ./data/memory-repo
  branch: main
  pull_interval_seconds: 30
  push_on_write: true
```

每轮对话会生成独立 JSON 快照，目录结构为：

```text
memory/projects/<project>/sessions/<session>/<timestamp>-<device>-<event>.json
```

请求前会按间隔拉取并导入远端快照；回答后会提交并推送当前会话。GitHub 或网络暂时失败时，对话仍正常返回，事件保留为 pending，后台会继续重试。可通过以下接口查看和手动同步：

```bash
curl http://127.0.0.1:8086/v1/memory/projects
curl http://127.0.0.1:8086/v1/memory/projects/basketball-live/sessions
curl -X POST http://127.0.0.1:8086/v1/memory/sync
curl http://127.0.0.1:8086/health
```

认证建议使用 SSH deploy key，或仅授权记忆仓库 Contents 读写权限的 fine-grained GitHub PAT 配合 credential helper。不要把 token 放进 Git URL、YAML 或仓库。多设备尽量使用不同 `session_id`；同一会话同时编辑时，目前按消息更多、时间更新的快照优先合并。

Git 适合个人低频协同、审计和备份，不适合高并发写入。未来扩展为多用户或多云实例时，应以 PostgreSQL 作为实时存储，以对象存储或 Git 作为归档层。

## 云端部署

服务器安装 Docker 后，创建 `config.yml`，将 `server.host` 设置为 `0.0.0.0`，开启 `git_sync`，然后通过环境变量注入密钥：

```bash
export DAMSELFISH_API_KEY='为客户端设置的路由访问密钥'
export DAMSELFISH_MEMORY_GIT_URL='git@github.com:你的账号/damselfish-memory.git'
export DAMSELFISH_DEVICE_ID='cloud-primary'
export AGNES_API_KEY='云模型密钥'
docker compose up -d --build
docker compose logs -f damselfish
```

`compose.yml` 使用 Docker volume 持久化 `/app/data`，其中包含 SQLite 数据库和记忆 Git 工作区。若记忆仓库使用 SSH，在生产环境通过 Docker secret 或只读 volume 将 deploy key 与 `known_hosts` 挂载到容器用户 `/home/damselfish/.ssh`；不要把私钥复制进镜像。

云端务必通过防火墙或 HTTPS 反向代理限制访问，并设置 `DAMSELFISH_API_KEY`。同一记忆仓库可被云端和多台客户端使用，但每台设备都应设置唯一的 `DAMSELFISH_DEVICE_ID`。

## 调度规则

1. 排除未配置密钥、已禁用和熔断中的目标。
2. 排除不具备必需能力的目标，例如工具请求只进入带 `tools` 标签的模型。
3. 用 EWMA 延迟、失败率、配置优先级、能力及人物匹配计算得分。
4. 按得分从低到高调用；任一目标失败会在同一请求内自动尝试下一个。
5. 后台探测更新空闲模型延迟，实际请求也持续更新评分。

`priority` 越低越优先。若只想严格选择最快模型，可将目标的 `priority` 设成相同值。主动探测会产生少量 API 调用；额度敏感的模型应设置 `probe: false`。

## Hermes 接入

把 Hermes 的 OpenAI 兼容 provider 指向：

```text
base_url: http://127.0.0.1:8086/v1
model: damselfish/auto
api_key: 任意值（未设置 DAMSELFISH_API_KEY 时）
```

如果设置了 `DAMSELFISH_API_KEY`，Hermes 必须使用同一密钥。通过 `curl http://127.0.0.1:8086/stats` 可查看每次路由、延迟、熔断和错误。

## 测试

```bash
uv sync --extra test
uv run pytest
```

## 云节点管理页面

部署后打开 `https://服务器/damselfish/admin/nodes`，输入服务器配置的
`DAMSELFISH_API_KEY`，即可添加和测试 OpenAI 兼容云节点。管理 API 与模型 API
使用相同的 Bearer Key 验证；上游 API Key 只写入服务器权限为 `0600` 的
`managed-nodes.json`，列表和编辑接口不会回显密钥。

页面支持获取上游模型列表、延迟测试、测试后保存、编辑和删除。节点保存后会热更新
模型池，无需重启 Damselfish。
