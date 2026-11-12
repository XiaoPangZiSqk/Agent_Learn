# S01 · Agent Loop（智能体循环）

用不到 110 行代码实现一个能真正执行 shell 命令的编码智能体：模型负责决策，程序负责执行，执行结果回填给模型，再决策——如此循环直到任务完成。本目录是整个 Harness 系列的第一课，只保留最本质的骨架，不含任何框架。

## 目录

- [1. 目录结构](#1-目录结构)
- [2. 快速开始](#2-快速开始)
- [3. 核心概念：Agent Loop](#3-核心概念agent-loop)
- [4. 关键技术讲解](#4-关键技术讲解)
  - [4.1 Responses API：与模型通信](#41-responses-api与模型通信)
  - [4.2 Function Calling：把工具交给模型](#42-function-calling把工具交给模型)
  - [4.3 工具结果回传协议](#43-工具结果回传协议)
  - [4.4 Shell 执行与安全边界](#44-shell-执行与安全边界)
  - [4.5 会话历史与上下文管理](#45-会话历史与上下文管理)
  - [4.6 配置管理：dotenv](#46-配置管理dotenv)
  - [4.7 命令行交互层](#47-命令行交互层)
- [5. 代码导读](#5-代码导读)
- [6. 已知局限与下一步](#6-已知局限与下一步)

## 1. 目录结构

```
S01_agent_loop/
├── agent_loop.py     # 全部实现：配置 + 工具定义 + 循环 + 交互式终端
└── README.md         # 本文档
```

| 名称 | 类型 | 主要作用 |
| --- | --- | --- |
| `agent_loop.py` | 文件 | 唯一源码文件。定义模型客户端、`bash` 工具、`run_bash()` 执行器、`agent_loop()` 主循环，以及 `__main__` 中的 REPL 入口。 |
| `README.md` | 文件 | 本目录的说明文档。 |
| `__pycache__/` | 目录（运行时生成） | Python 导入模块时自动生成的字节码缓存，可安全删除，不属于源码。 |

项目根目录还有一个对本目录至关重要的文件：

| 名称 | 位置 | 作用 |
| --- | --- | --- |
| `.env` | 项目根 `Agent_Learn/` | 存放 `DS_API_KEY`、`DS_BASE_URL`、`DS_MODEL`。被 `load_dotenv()` 自动向上查找加载，不随代码分发。 |

## 2. 快速开始

**依赖**

```bash
pip install openai python-dotenv
```

**环境变量**（写在项目根 `.env`）

```dotenv
DS_API_KEY=<你的 API Key>
DS_BASE_URL=https://api.deepseek.com
DS_MODEL=deepseek-flash
```

**运行**

```bash
cd Harness/S01_agent_loop
python agent_loop.py
```

输入自然语言任务即可，例如 `统计当前目录下 .py 文件的行数`；输入 `q`、`exit` 或直接回车退出。

**执行流程示例**

```
s01 >> 统计当前目录下 .py 文件的行数
第 0 次请求
$ ls -la
$ wc -l *.py
当前目录下共有 1 个 .py 文件，总行数 100 行
```

## 3. 核心概念：Agent Loop

普通聊天机器人是**一问一答**；智能体是**一问、多轮自主行动、一答**。二者差的正是这个循环：

```
用户任务
   ↓
[模型推理] ── 不需要工具 ──→ 直接输出答案，结束
   ↓ 需要工具
[生成 tool call] → [程序执行] → [结果回填] → 再次请求模型
   ↑                                              │
   └──────────────────────────────────────────────┘
```

这个模式源自 ReAct（Reasoning + Acting）：模型每一步都先"想"再"做"，把外部世界的真实反馈当作下一步推理的输入。代码中对应五个动作，构成一次迭代：

| 步骤 | 代码 | 说明 |
| --- | --- | --- |
| 1. 请求 | `client.responses.create(...)` | 把历史 + 可用工具交给模型 |
| 2. 记账 | `messages.extend(response.output)` | 本轮的推理、文本、工具调用原样入库 |
| 3. 判定 | `if not tool_calls: return` | 没有工具调用即任务收敛 |
| 4. 执行 | `run_bash(command)` | 真实副作用发生在这里 |
| 5. 回填 | `messages.append({"type": "function_call_output", ...})` | 把观察结果接回上下文，进入下一轮 |

**关键的循环不变式**：只要模型还在要工具，就一定不能退出循环，也一定不能只回填一半。

## 4. 关键技术讲解

### 4.1 Responses API：与模型通信

本项目使用 OpenAI SDK 的 **Responses API**（`client.responses.create`），它与传统的 Chat Completions（`chat.completions.create`）是两代接口。

| 概念 | Chat Completions | Responses API |
| --- | --- | --- |
| 消息载体 | `messages` 列表，纯字符串/字典 | `input` 列表，可混入推理项、工具调用项等结构化对象 |
| 系统提示 | `messages[0]` 中的 `system` 角色 | 独立的 `instructions` 参数 |
| 输出 | `choices[0].message` 单条 | `output` 列表，可能同时含 `reasoning`、`message`、`function_call` |
| 工具定义 | 需包一层 `{"type":"function","function":{...}}` | `type`、`name`、`description`、`parameters` 直接平铺在顶层 |

本项目的调用参数：

```python
response = client.responses.create(
    model=MODEL,                # 由环境变量注入，换模型不改代码
    instructions=SYSTEM,        # 系统提示：角色 + 工作目录 + 行为约束
    input=messages,             # 完整对话历史，跨轮持续累积
    tools=TOOLS,                # 可用工具清单
    tool_choice="auto",         # 由模型自行决定是否调用、调用哪个
    max_output_tokens=8000,     # 单次回复上限，防止跑飞
)
```

`tool_choice` 的常见取值：`"auto"`（模型自决）、`"none"`（禁用）、`"required"`（强制至少调一个）、或指定具体函数名。本课用 `auto`，因为"要不要行动"本身就该由模型判断。

### 4.2 Function Calling：把工具交给模型

模型不能直接跑 shell，它只能"申请"跑 shell。所谓 Function Calling，就是**用 JSON Schema 向模型描述工具签名**，模型返回结构化的调用请求，由宿主程序执行。

```python
TOOLS = [{
    "type": "function",
    "name": "bash",
    "description": "运行一个shell命令",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,
    },
    "strict": True,
}]
```

三个要点：

1. **`description` 是给模型看的文档**，不是注释。描述越准确，模型选错工具、拼错参数的概率越低。
2. **`parameters` 是标准 JSON Schema**，模型据此生成合法 JSON。这里只有 `command` 一个字符串参数，但换成多参数工具写法不变。
3. **`strict: true` 开启严格模式**，要求服务端按 schema 约束输出，避免少字段、多字段、类型错。严格模式有硬性前提：对象必须显式声明 `additionalProperties: false`，且 `properties` 中的字段要全部列入 `required`。代码里那行注释说的就是这个坑——漏掉会导致请求被拒。

模型返回的调用请求形如：

```json
{
  "type": "function_call",
  "name": "bash",
  "call_id": "call_00_ET_CzkOT3HD0kn5aQNNISPy6103",
  "arguments": "{\"command\": \"ls -la\"}"
}
```

注意 `arguments` 是**字符串**而不是对象，必须 `json.loads` 后取字段。模型生成的 JSON 不保证是合法 JSON 文本，生产环境应加 try/except 兜底。

### 4.3 工具结果回传协议

这是最容易写错的地方。回填不是拼一段文字给模型看，而是发一条**结构化应答项**：

```python
messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,   # 必须与请求方的 call_id 严格相等
    "output": output,          # 字符串；结构化结果请自行 json.dumps
})
```

两条硬性规则：

- **`call_id` 必须一一配对。** 服务端校验 `function_call_output` 能否找到对应的 `function_call`，找不到直接 400：`No tool call found for tool output with call_id ...`。
- **`function_call` 项本身必须写回历史。** 这也是 `messages.extend(response.output)` 存在的理由：把 `reasoning`、`message`、`function_call` 整批原样接回去。若只回填 output 而不回写调用项，同样触发上面的 400；若把 `output` 拆成纯文本 `{"role":"assistant","content":...}`，则丢失 `call_id` 语义，同样失败。

`response.output` 中可能出现三类项，均有用途：

| 类型 | 含义 | 本项目的处理 |
| --- | --- | --- |
| `reasoning` | 模型的思考摘要与加密内容 | 原样回写，保留推理上下文 |
| `message` | 面向用户的最终文本 | 通过 `response.output_text` 读取并作为返回值 |
| `function_call` | 工具调用请求 | 解析 `arguments` 并执行 |

**一轮可能返回多个调用。** 模型可以同时申请 `ls -la`、`find ...`、创建文件等多个命令。因此代码先收集成 `tool_calls` 列表再逐个执行，确保每个 `call_id` 都有应答；漏一个，下一轮请求就会失败。

### 4.4 Shell 执行与安全边界

`run_bash()` 是唯一产生真实副作用的地方，也是风险最集中的地方。

```python
r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                   capture_output=True, text=True, errors="replace", timeout=120)
```

| 参数 | 作用 | 为什么必须有 |
| --- | --- | --- |
| `shell=True` | 交给 `/bin/sh` 解析 | 让模型能用管道、重定向、`&&` 等完整 shell 语法 |
| `capture_output=True` | 捕获 stdout/stderr | 结果要作为观察回传给模型，不能直接刷屏 |
| `text=True` + `errors="replace"` | 以文本解码，非法字节替换 | 防止二进制输出导致 `UnicodeDecodeError` 中断循环 |
| `timeout=120` | 超时强杀 | 防止 `sleep`、交互式命令把智能体永久卡死 |
| `cwd=os.getcwd()` | 固定工作目录 | 与 `SYSTEM` 中告知模型的工作目录保持一致 |

配套的两道防线：

1. **危险命令黑名单**：命中 `rm -rf /`、`sudo`、`shutdown`、`reboot`、`> /dev/` 直接返回错误串而不执行。注意这只是子串匹配，属于教学演示级别——`rm  -rf /`（双空格）或 `$(echo sudo)` 都能绕过，不可用于生产。
2. **输出截断**：`out[:50000]` 限制单次观察的长度。命令输出动辄上万行，全量塞回上下文会迅速耗尽 token 预算；截断是廉价的上下文保护。

### 4.5 会话历史与上下文管理

`messages` 是一个**只增不减**的列表，跨用户提问持续累积，类型上混合了普通字典（`role` 消息）与 SDK 返回的 pydantic 对象（`reasoning`、`function_call`、`message`），SDK 能正确序列化两者。

```python
history.append({"role": "user", "content": query})   # 用户输入
messages.extend(response.output)                     # 模型输出项
messages.append({"type": "function_call_output", ...})  # 工具结果
```

这样做的收益是**多轮记忆**：第二个问题能引用第一个问题执行过的结果。代价是**上下文线性膨胀**，最终会撞上模型窗口上限，并让每次请求更慢更贵。

`MAX_TURNS = 20` 是另一层保护：限制单次任务的最大迭代次数，避免模型反复调用同一个失败命令形成死循环。它和 `timeout` 的分工是——前者防"轮次爆炸"，后者防"单条命令卡死"。

### 4.6 配置管理：dotenv

```python
load_dotenv()                                              # 从 .env 读入 os.environ
API_KEY = os.getenv("DS_API_KEY")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
```

要点：

- **密钥不进代码、不进版本库**。`.env` 由 `.gitignore` 排除，代码只读环境变量名。
- **`base_url` 决定服务商**。DeepSeek 兼容 OpenAI 协议，因此可以用同一个 SDK，只换地址与模型名；`responses` 与 `chat.completions` 两个端点在该服务上均可用。
- **模型名外置**，方便在同一份代码上对比不同模型的表现。

### 4.7 命令行交互层

`__main__` 是一个极简 REPL：读一行、追加到历史、调 `agent_loop`、打印结果。

```python
query = input("\001\033[36m\002s01 >> \001\033[0m\002")
```

其中 `\033[36m` 是 ANSI 转义序列（青色前景），`\001` 与 `\002` 是 Readline 的标记，告诉它"这两个字符不占显示宽度"。没有这对标记时，长命令按方向键回退会出现错位。`EOFError` 处理 `Ctrl+D`，`KeyboardInterrupt` 处理 `Ctrl+C`，保证能干净退出。

## 5. 代码导读

按职责从上到下分四块：

| 区块 | 内容 | 职责 |
| --- | --- | --- |
| 配置区 | `load_dotenv()`、`API_KEY`/`BASE_URL`/`MODEL`、`client`、`SYSTEM`、`TOOLS`、`MAX_TURNS` | 建立客户端、定义身份与工具契约 |
| `run_bash(command)` | 黑名单校验 → 子进程执行 → 合并 stdout/stderr → 截断 | 唯一的副作用入口，任何失败都返回字符串而非抛异常 |
| `agent_loop(messages)` | 请求 → 写回 → 判定 → 执行 → 回填，最多 `MAX_TURNS` 轮 | 循环主体，返回最终文本 |
| `if __name__ == "__main__"` | REPL、历史维护、结果打印 | 交互入口 |

设计上值得注意的三点：

1. **`run_bash` 永不抛异常**，所有错误都转成字符串回传。因为异常会直接打断循环，而模型看到 `Error: Timeout (120s)` 反而有机会自己换个命令重试。
2. **`agent_loop` 接收历史列表而不是返回新列表**，原地追加，让调用方自然获得多轮上下文。
3. **打印调试信息**（`第 i 次请求`、`response.output`）保留在循环内，方便观察每一轮的原始结构；正式使用时可以删掉或改为日志。

## 6. 已知局限与下一步

当前实现刻意保持最小，以下都是可预见的改造点：

| 局限 | 影响 | 改进方向 |
| --- | --- | --- |
| 黑名单子串匹配 | 易绕过，非安全边界 | 命令白名单、沙箱、容器隔离 |
| 历史无限增长 | 长会话超出上下文窗口 | 摘要压缩、滑动窗口、丢弃旧工具输出 |
| 工具只有一个 `bash` | 所有能力挤在字符串里，难以复用与审计 | 拆成 `read_file`、`write_file`、`edit_file`、`search` 等结构化工具（S02 主题） |
| 串行执行 | 多个独立命令依次等待 | 只读命令并行执行 |
| 无持久化 | 退出即丢，无法回放 | 会话落盘、trace 记录 |
| 无人工确认 | 危险操作直接执行 | 写操作前请求用户批准（human-in-the-loop） |

理解了这百来行，后面所有复杂度都只是它的展开：更多工具、更好的提示词、更强的上下文管理，循环本身不会变。
