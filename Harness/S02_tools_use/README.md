# S02 · Tool Use（结构化工具集）

在 S01 的循环骨架上做一件事：把"万事皆 bash"拆成 5 个带参数契约的结构化工具，让模型的每个动作可描述、可校验、可审计。循环本身没有变化，变的是工具层的设计方式。

## 目录

- [1. 目录结构](#1-目录结构)
- [2. 快速开始](#2-快速开始)
- [3. 从 S01 到 S02：为什么要拆工具](#3-从-s01-到-s02为什么要拆工具)
- [4. 关键技术讲解](#4-关键技术讲解)
  - [4.1 工具集设计](#41-工具集设计)
  - [4.2 工具契约：Schema 与函数签名](#42-工具契约schema-与函数签名)
  - [4.3 工具分发：注册表模式](#43-工具分发注册表模式)
  - [4.4 strict 模式与可选参数](#44-strict-模式与可选参数)
  - [4.5 路径安全：防逃逸](#45-路径安全防逃逸)
  - [4.6 文件三原语：读、写、改](#46-文件三原语读写改)
  - [4.7 glob：按模式搜索](#47-glob按模式搜索)
  - [4.8 结果回传与错误即数据](#48-结果回传与错误即数据)
- [5. 代码导读](#5-代码导读)
- [6. 实测行为](#6-实测行为)
- [7. 已知局限与下一步](#7-已知局限与下一步)

## 1. 目录结构

```
S02_tools_use/
├── tool_use.py       # 全部实现：5 个工具 + 分发表 + Agent 循环 + 终端入口
└── README.md         # 本文档
```

| 名称 | 类型 | 主要作用 |
| --- | --- | --- |
| `tool_use.py` | 文件 | 唯一源码文件。含配置区、`run_bash`（沿用 S01）、`safe_path` 路径守卫、`run_read` / `run_write` / `run_edit` / `run_glob` 四个文件工具、`TOOLS` 声明、`TOOL_HANDLERS` 分发表、`agent_loop()` 循环、`__main__` REPL。 |
| `README.md` | 文件 | 本目录的说明文档。 |
| `__pycache__/` | 目录（运行时生成） | 导入模块产生的字节码缓存，可安全删除。 |

外部依赖：

| 名称 | 位置 | 作用 |
| --- | --- | --- |
| `.env` | 项目根 `Agent_Learn/` | 提供 `DS_API_KEY`、`DS_BASE_URL`、`DS_MODEL`。 |

## 2. 快速开始

**依赖**

```bash
pip install openai python-dotenv
```

**环境变量**（项目根 `.env`）

```dotenv
DS_API_KEY=<你的 API Key>
DS_BASE_URL=https://api.deepseek.com
DS_MODEL=deepseek-flash
```

**运行**

```bash
cd Harness/S02_tools_use
python tool_use.py
```

输入 `q`、`exit` 或空行退出。

**注意工作区范围**：`WORKDIR = Path.cwd()` 取的是**启动时的工作目录**，不是脚本所在目录。从 `S02_tools_use/` 启动，文件类工具就只能访问该目录及其子目录；若从项目根启动，则整个项目都是可写范围。

**典型交互**

```
s02 >> 把 README 里第一段标题改成 S02
第 0 次请求
> glob
$ {'pattern': '**/README.md'}
README.md
> read_file
$ {'path': 'README.md', 'limit': 5}
...
> edit_file
$ {'path': 'README.md', 'old_text': '# S02 · Tool Use（结构化工具集）', 'new_text': '# S02 · Tool Use'}
Edited README.md
已修改标题。
```

## 3. 从 S01 到 S02：为什么要拆工具

S01 只有一个 `bash` 工具，所有能力都压在一个字符串参数里。这在演示阶段足够，但一旦进入真实使用就会暴露四个问题：

| 问题 | 表现 | 拆分后如何缓解 |
| --- | --- | --- |
| 不可审计 | 一条命令里串联 `cd`、`ls`、`find`、`touch`，日志无法归因到底做了什么 | 每个动作一次调用，`name` + `args` 就是结构化日志 |
| 无法校验 | 路径、行数等参数藏在命令文本里，程序无从检查 | 参数进 JSON Schema，类型与必填项先由 schema 把关 |
| 难以授权 | 给了 bash 就等于给了整台机器，粒度只能"全开或全关" | 可按工具授权，`glob` 可放开、`write_file` 可收紧 |
| 结果噪声 | 整段 shell 输出进上下文，几十 KB 的 `ls -lR` 直接吃掉预算 | 每个工具自行裁剪返回内容 |

S02 的工具分工：

| 工具 | 职责 | 是否写副作用 |
| --- | --- | --- |
| `bash` | 兜底：任何未封装的操作 | 是（不受路径守卫约束） |
| `read_file` | 读文本文件，可限制行数 | 否 |
| `write_file` | 写文件，自动建父目录 | 是 |
| `edit_file` | 对已有文件做定点文本替换 | 是 |
| `glob` | 按通配符查找文件路径 | 否 |

关键区别在于**能力边界从"字符串"变成了"函数签名"**：模型不再描述怎么做，而是说明做什么，执行细节由程序掌控。

## 4. 关键技术讲解

### 4.1 工具集设计

工具集设计的第一原则是**按使用者（模型）的意图划分，而不是按系统调用划分**。模型想的是"我要看看有哪些文件""我要把这段改掉"，对应 `glob` 和 `edit_file`；如果没有这些工具，它只能构思出 `ls` 和 `sed` 命令，把意图翻译成 shell 语法的负担就落在模型身上，出错率随之上升。

三条实践准则：

1. **一个工具只做一类事**。`edit_file` 不兼任创建文件，`write_file` 不兼任读。
2. **参数要显式**。`run_edit` 用 `old_text`/`new_text` 两个字段表达"从什么改成什么"，比让模型拼 `sed -i` 安全得多。
3. **返回值要收敛**。返回值是给模型看的观察，不是给终端看的日志，必须控制长度。

但拆分不是越多越好：工具数量增加会抬高选择成本，模型更容易选错工具。实践中常见做法是"少量正交工具 + 一个 bash 兜底"，本文件正是这个结构。

### 4.2 工具契约：Schema 与函数签名

每个工具在代码里存在两份定义，必须严格对齐：

| 位置 | 内容 | 面向谁 |
| --- | --- | --- |
| `TOOLS` | `name`、`description`、`parameters`（JSON Schema） | 模型 |
| `run_*` 函数 | 真实的 Python 函数名与形参 | 解释器 |

对齐关系是硬性的：

```python
handler = TOOL_HANDLERS.get(call.name)      # ① 名字查表
args = json.loads(call.arguments or "{}")
output = handler(**args)                     # ② 字典解包成关键字参数
```

- ① 要求 `TOOLS` 里的 `name` 与 `TOOL_HANDLERS` 的键一致。
- ② 要求 schema 的 `properties` 键名与函数形参名逐字相同。`read_file` 的 schema 声明 `path`、`limit`，函数就必须写成 `run_read(path, limit=None)`，写成 `file_path` 会直接 `TypeError`。

`description` 同样是契约的一部分，它决定模型何时选这个工具。`"在指定工作区中读取文件内容"` 里的"工作区"三个字并非客套——它在提示模型路径应当写相对路径。

**规范建议**：把 schema 与函数放在相邻位置（本文件把 `run_*` 与 `TOOLS` 分成两段，中间隔了几十行），或者用装饰器从函数签名自动生成 schema，避免改一处漏一处。

### 4.3 工具分发：注册表模式

```python
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}
```

这叫**注册表（registry）/ 分发表**模式，作用是切断"工具数量"与"循环代码"的耦合：

- 新增工具 = `TOOLS` 加一项 + `TOOL_HANDLERS` 加一项，`agent_loop` 一行不改。
- 若写成 `if call.name == "bash": ... elif call.name == "read_file": ...`，每加一个工具都要改循环本体，且容易漏分支。

`handler(**args)` 的字典解包是分发能成立的基础：模型返回的 `arguments` 是 JSON 对象，解包后恰好对应函数的关键字参数。

安全细节：

```python
output = handler(**args) if handler else f"Unknown: {call.name}"
```

未知工具名不抛异常，而是返回一条错误字符串交给模型。模型下一轮看到 `Unknown: xxx` 会自己换一个工具——这是"错误即数据"理念的具体落地。

### 4.4 strict 模式与可选参数

`strict: true` 让服务端按 schema 约束生成，避免少字段、多字段、类型错。它有两个硬性前提：

1. 每个对象必须显式声明 `"additionalProperties": false`。
2. **`properties` 里的所有字段都必须出现在 `required` 中。**

第 2 条常被忽略，而本文件恰好踩在边线上：

```python
"parameters": {"type": "object",
               "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
               "required": ["path"], "additionalProperties": False}
```

`limit` 是可选参数，没有列入 `required`。按 OpenAI 官方规范，这样的 schema 在严格模式下会被拒绝；实测 DeepSeek 端点接受这种写法，`limit` 也能正常作为可选参数使用——但**换服务商或换模型时可能直接 400**，属于可移植性隐患。

两种正规修法：

- **全必填 + 允许 null**：把 `limit` 加进 `required`，类型写成 `["integer", "null"]`，模型不需要时传 `null`，函数侧 `limit: int|None = None` 正好接住。
- **该工具关掉 strict**：对参数简单的工具影响不大，但失去服务端校验。

对应的 Python 侧写法 `def run_read(path: str, limit: int|None = None)` 已经兼容 `null`，这是接口设计与实现细节配合的一处示范。

### 4.5 路径安全：防逃逸

`read_file` 让模型传路径，等于把"访问哪个文件"的决定权交给模型。最直接的攻击是：

```
../../etc/passwd
```

这就是**路径逃逸（path traversal）**。仅靠字符串检查很难穷尽，正确做法是先把路径规范化，再判断归属：

```python
path = (WORKDIR / sub_path).resolve()      # 规范化：消掉 . / .. / 符号链接
if not path.is_relative_to(WORKDIR):       # 归属判断：是否仍在工作区内
    raise ValueError(f"Path escapes workspace: {sub_path}")
return path
```

两个 API 的作用：

| API | 作用 | 为什么不能用替代方案 |
| --- | --- | --- |
| `Path.resolve()` | 转绝对路径并折叠 `.`、`..`、符号链接 | 不规范化就无法判断真实位置，`a/../b` 这类写法会绕过 |
| `Path.is_relative_to()` | 判断路径是否位于某目录之下 | 不能用 `str.startswith`：`/work_evil` 会误判为在 `/work` 之下 |

实测拦截结果（从本目录启动）：

| 传入 | 结果 |
| --- | --- |
| `README.md` | 放行，解析为 `<WORKDIR>/README.md` |
| `../S01_agent_loop/agent_loop.py` | 拦截：`Path escapes workspace` |
| `/etc/passwd` | 拦截（绝对路径拼接后仍在工作区外） |
| `a/../../x` | 拦截（`..` 规范化后越界） |

**三个必须知道的局限**：

1. **它只保护文件类工具，管不了 `bash`**。`cat /etc/passwd` 依然可以读系统文件——安全策略出现缺口，攻击者只需要选一个不受约束的工具。这是当前设计最明显的不一致。
2. **检查与使用之间存在时间差（TOCTOU）**：`resolve()` 判定通过后、真正读写前，若路径上的符号链接被替换，仍可能越界。真正的防护要靠容器 / 只读挂载 / 系统级权限。
3. **工作区边界由启动目录决定**：`WORKDIR = Path.cwd()`。若从 `/` 启动，整个文件系统都成了"工作区"，守卫形同虚设。生产实现通常用脚本所在目录或显式配置项。

### 4.6 文件三原语：读、写、改

**`run_read(path, limit)`**

```python
lines = safe_path(path).read_text(encoding="utf-8").splitlines()
if limit and limit < len(lines):
    lines = lines[:limit] + [f"... (余 {len(lines)-limit}) 行"]
```

- `splitlines()` 按行切分，`limit` 用来控制上下文体积。
- 截断时**保留一句剩余行数提示**，让模型知道内容不全，避免它基于残缺内容下结论。
- 小瑕疵：`limit=0` 会被 `if limit` 当作假值，等于不截断。要严格的话应写 `if limit is not None`。
- 返回内容**不带行号**。这对 `edit_file` 反而是好事：模型复制的 `old_text` 不会混入行号前缀。

**`run_write(path, content)`**

```python
file_path.parent.mkdir(parents=True, exist_ok=True)
file_path.write_text(content, encoding="utf-8")
```

- `parents=True` 让 `a/b/c.txt` 这种多级路径自动建目录，`exist_ok=True` 让重复写入不报错，整体是**幂等**的。
- 语义是**整文件覆盖**：模型只需要提供最终内容，不必先读。这对新建文件最友好，对已存在文件则是高风险操作。

**`run_edit(path, old_text, new_text)`**

```python
text = file_path.read_text(encoding="utf-8")
if old_text not in text:
    return f"Error: text not found in {path}"        # 未命中 -> 返回错误串，不改文件
file_path.write_text(text.replace(old_text, new_text, 1), ...)
```

- 采用**读—改—写**：先校验锚点存在，再替换。
- `replace(..., 1)` 只替换第一处。若 `old_text` 在文件中出现多次，可能改到非预期的位置，而结果看起来"成功"了。稳妥做法是先统计出现次数，`count != 1` 就要求模型补充更多上下文。
- 不命中时返回 `Error: text not found`，模型通常会重新读文件、换一段更长的锚点重试——这是把错误当观察喂回去的典型收益。
- 非原子写入：直接 `write_text` 覆盖，进程中途被杀会留下半截文件。生产实现应写临时文件再 `os.replace()`。
- 为什么不让模型"读全文 + 写全文"来改？那会消耗双倍 token，且一旦模型在重写时漏掉几行，损失的是整个文件。定点替换把影响面限制在局部。

### 4.7 glob：按模式搜索

```python
matches = sorted({
    match for match in g.glob(pattern, root_dir=WORKDIR, recursive=True)
    if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
})
```

逐层看：

| 要素 | 作用 |
| --- | --- |
| `root_dir=WORKDIR` | 让模式相对工作区匹配，模型可以写 `**/*.py` 而不用写绝对路径 |
| `recursive=True` | 开启 `**` 递归匹配；不设则 `**` 只当普通字符 |
| `set` 推导式 | 去重：`**/*.py` 与 `*.py` 等模式可能产生重复路径 |
| `sorted()` | 输出稳定，便于模型比对上轮结果，也便于人读 |
| `resolve().is_relative_to()` | 过滤经符号链接逃出工作区的匹配项 |
| `matches[:200]` + 省略提示 | 收敛上下文体积，同时告知模型结果被截断 |

注意 `glob` **不区分文件与目录**（`README.md` 会匹配，`Harness` 这样的目录也会匹配），也不看文件内容。定位到路径后，模型需要接一次 `read_file` 才能真正读到内容。若需要按内容筛选，必须另加 `grep` 类工具。

### 4.8 结果回传与错误即数据

所有 handler 都满足同一约定：**入参是模型给的 JSON，出参是字符串，且永不抛异常**。

```python
def run_read(path, limit=None) -> str:
    try:
        ...
    except Exception as e:
        return f"error : {e}"
```

为什么统一返回字符串而不是抛异常？

1. **异常会打断循环**：一旦抛出，模型看不到任何反馈，任务直接失败。
2. **字符串是可被推理的观察**：`Error: text not found in README.md` 会引导模型改变策略；异常堆栈对模型既难读又无用。
3. **异常串也是日志**：`Error: Path escapes workspace: ../../etc/passwd` 同时具备安全告警的价值。

回填仍遵循 S01 的协议：`function_call_output` 必须按 `call_id` 与调用项配对，且一轮内的多个调用必须**全部**应答，漏一个下一轮就会 400。

一处与 S01 的差异值得注意：本文件用 `print(output)` 打印完整结果，而 S01 是 `print(output[:200])`。读取大文件时会刷屏，调试时建议改回截断打印。

## 5. 代码导读

| 区块 | 内容 | 职责 |
| --- | --- | --- |
| 配置区 | `load_dotenv()`、`WORKDIR`、`client`、`SYSTEM` | 建立客户端与工作区边界 |
| `run_bash` | 黑名单 → `subprocess.run` → 截断 | 沿用 S01 的兜底执行器 |
| `safe_path` | `resolve()` + `is_relative_to()` | 所有文件类工具的统一入口守卫 |
| `run_read` / `run_write` / `run_edit` / `run_glob` | 文件操作原语 | 每个都是"try 包裹 + 返回字符串"的独立单元 |
| `TOOLS` | 5 个函数的 JSON Schema | 给模型看的工具清单 |
| `TOOL_HANDLERS` | 名称 → 函数的映射 | 分发注册表，与 `TOOLS` 一一对应 |
| `agent_loop` | 请求 → 写回 → 判定 → 分发执行 → 回填，最多 `MAX_TURNS` 轮 | 循环主体 |
| `__main__` | REPL + 历史维护 | 交互入口 |

三处设计取舍：

1. **`raise` 与 `return` 混用有明确分工**：`safe_path` 抛 `ValueError`（内部错误），各 `run_*` 用 `try/except Exception` 接住并转成字符串（对外契约），因此模型看到的永远是文案而非堆栈。
2. **守卫集中在一处**：四个文件工具全部经 `safe_path`，加密钥逻辑时只需改一个函数。
3. **循环与 S01 几乎一致**：唯一的改动是 `run_bash(...)` 换成 `handler(**args)`，这正说明把工具分发抽象出来的价值。

## 6. 实测行为

以下为从本目录启动、使用 `deepseek-flash` 的真实观测：

**工具定义被接受**。5 个工具连同 `strict: true` 一起下发无报错，说明 DeepSeek 对 `limit` 未列入 `required` 的写法宽容。

**模型会自行组合工具**。任务"读取 tool_use.py 的前 10 行"时，模型第一轮同时发出两个调用：

```
> bash   {"command": "ls /home/zleb/project/python/Agent_Learn/Harness/S02_tools_use"}
> glob   {"pattern": "**/tool_use.py"}
```

定位文件后才发 `read_file`。这说明拆分工具没有削弱能力，反而让"先定位再读取"的意图变得可见——在 S01 里，这些步骤会挤成一条命令。

**守卫按预期拦截**：`../S01_agent_loop/agent_loop.py`、`/etc/passwd`、`a/../../x` 三者全部返回 `Path escapes workspace`。

**注册表完整**：`{t["name"] for t in TOOLS}` 与 `TOOL_HANDLERS.keys()` 完全相等。

## 7. 已知局限与下一步

| 局限 | 影响 | 改进方向 |
| --- | --- | --- |
| `bash` 不受 `safe_path` 约束 | 文件守卫可被绕过 | 要么禁用 bash，要么在容器中运行整个进程 |
| `handler(**args)` 未捕获 `TypeError` | schema 与函数签名不一致时循环直接崩溃 | 分发处 `try/except TypeError`，转成错误字符串 |
| `strict` + 可选参数 | 换服务商可能 400 | 全必填 + `null`，或该工具关 strict |
| `edit_file` 非原子、不校验唯一性 | 可能改错位置或写出半截文件 | 统计锚点出现次数；临时文件 + `os.replace()` |
| 无内容检索 | 只能按文件名找，找不到"哪里调用了这个函数" | 增加 `grep` 工具 |
| 无写操作确认 | `write_file` 可覆盖任意工作区文件 | 写前请求用户批准（human-in-the-loop） |
| 结果全量打印 | 大文件刷屏 | 打印截断 + 落日志文件 |
| 历史无限增长 | 长会话超上下文窗口 | 摘要压缩、旧工具输出裁剪 |

下一步的自然方向有两条：补齐 `grep` 与目录遍历，让"搜索—阅读—修改"闭环完整；或者引入权限分级与人工确认，把"能做什么"从提示词约束升级为程序约束。
