"""
解决 S01 工具只有一个 bash	所有能力挤在字符串里，难以复用与审计的局限
	拆成 read_file、write_file、edit_file、search 等结构化工具
"""

import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# 获取当前工作目录 -- /Agent_Learn/
WORKDIR = Path.cwd()

# 使用deepseek作为底座大模型
API_KEY = os.getenv("DS_API_KEY")
BASE_URL = os.getenv("DS_BASE_URL")
MODEL = os.getenv("DS_MODEL")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM = f"你是 {WORKDIR} 的编码智能体。用 bash 来解决任务。行动，不要解释。"


# -----------------s01 tool
def run_bash(command: str) -> str:
    """
    运行系统 bash
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

# -------------------s02 tool
def safe_path(sub_path: str) -> Path :
    """
    判断path是否为安全路径，防止模型逃出工作目录。

    例如：模型（或被恶意文件诱导的提示注入）可能让 read_file 传入这样的 path：../../etc/passwd
    经过 resolve() 后的 path 可能在系统 etc/passwd 下，模型就能读到系统密码文件——这就是路径逃逸。
    is_relative_to() 可验证 path 是否在 WORKDIR 下，确保 path 在指定环境中

    局限：它防的是文件类工具；run_bash 走 shell，不受此约束（cat /etc/passwd 照样能读），这是之前提到的防护不一致之处。
    """
    path = (WORKDIR / sub_path).resolve()
    if not path.is_relative_to(WORKDIR) :
        raise ValueError(f"Path escapes workspace: {sub_path}")
    return path

def run_read(path: str, limit: int|None = None) -> str:
    """
    在安全路径下读取文件

    param
        path: WORKDIR 下的目录
        limit: 读取文件的行数
    """
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines) :
            lines = lines[:limit] + [f"... (余 {len(lines)-limit}) 行"]
        return "\n".join(lines)
    except Exception as e:
        return f"error : {e}"

def run_write(path: str, content: str) -> str:
    """
    写入文件
    """
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    编辑已存在的文件
    """
    try:
        file_path = safe_path(path)
        text = file_path.read_text(encoding="utf-8")
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    """
    按 glob 通配符模式查找文件

    局限：只列文件名，不看内容。模型通常 glob 定位后接 read_file 阅读，两工具配合使用。
    局限：不做文件类型过滤，目录也会出现在结果中（glob 不区分文件/目录）。
    """
    import glob as g
    try:
        matches = sorted({
            match for match in g.glob(
                pattern, root_dir=WORKDIR, recursive=True)
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })
        shown = matches[:200]
        if len(matches) > 200:
            shown.append("... (更多匹配被省略)")
        return "\n".join(shown) if shown else "(无匹配)"
    except Exception as e:
        return f"Error: {e}"

TOOLS = [
    {"type": "function",
    "name": "bash",                     
    "description": "运行一个shell命令",
    "parameters": { "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False, },
    "strict":True },
    {"type": "function", 
    "name": "read_file", 
    "description": "在指定工作区中读取文件内容",
    "parameters": { "type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"], "additionalProperties": False, },
    "strict":True },
    {"type": "function",
    "name": "write_file",                       
    "description": "在指定工作区中写入文件内容",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False, },
    "strict":True },
    {
    "type": "function",
    "name": "edit_file",                       
    "description": "对指定工作区中已有文件进行修改",
    "parameters": { "type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"], "additionalProperties": False, }, 
    "strict":True },
    {
    "type": "function",
    "name": "glob",                       
    "description": "按 glob 通配符模式查找指定工作区中的文件",
    "parameters": { "type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"], "additionalProperties": False, },
    "strict":True }
 ]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}

MAX_TURNS = 20  # 单轮对话最多迭代次数，防止死循环


def agent_loop(messages: list) -> str:
    """循环调用模型，直到它不再请求工具；返回最终文本回复。"""
    for i in range(MAX_TURNS):
        print(f"第 {i} 次请求\n")
        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM,
            input=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_output_tokens=8000,
        )
        # 本轮 assistant 输出（reasoning / message / function_call）要原样写回历史
        messages.extend(response.output)
        #print(f"response.output: {response.output} \n")

        # 为什么要先收集所有的函数调用？
        # 模型在一个 turn 里可以并发请求多个命令，如 ：新增一个文件：
        #   1. 先确定当前目录并列出文件 ls -la
        #   2. find 命令，查找文件
        #   3. 创建文件的命令。
        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:  # 没有工具调用 => 任务结束
            return response.output_text

        for call in tool_calls:
            handler = TOOL_HANDLERS.get(call.name)
            args = json.loads(call.arguments or "{}")
            print(f"\033[33m> {call.name}\033[0m")
            print(f"\033[33m$ {args}\033[0m")
            output = handler(**args) if handler else f"Unknown: {call.name}"
            print(output)
            # 用 function_call_output 按 call_id 把结果回传给模型
            messages.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            })

    return "Error: 超过最大迭代次数"


if __name__ == "__main__":
    history = []
    while True:
        try:
            # \001/\002 tell Readline the ANSI escapes have zero display width.
            query = input("\001\033[36m\002s01 >> \001\033[0m\002")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        print(agent_loop(history))
        print()