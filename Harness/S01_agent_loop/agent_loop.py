"""
演示 AI 编程智能体最核心、最本质的模式——Agent Loop（智能体循环）
"""

import json
import os
import subprocess

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# 使用deepseek作为底座大模型
API_KEY = os.getenv("DS_API_KEY")
BASE_URL = os.getenv("DS_BASE_URL")
MODEL = os.getenv("DS_MODEL")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM = f"你是 {os.getcwd()} 的编码智能体。用 bash 来解决任务。行动，不要解释。"

TOOLS = [{
    "type": "function",
    "name": "bash",                       # ← 直接是顶层字段
    "description": "运行一个shell命令",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
        "additionalProperties": False,   # strict 模式必须显式关闭
    },
    "strict":True   # 强调参数传递要规范。
}]


def run_bash(command: str) -> str:
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
        print(f"response.output: {response.output} \n")

        # 为什么要先收集所有的函数调用？
        # 模型在一个 turn 里可以并发请求多个命令，如 ：新增一个文件：
        #   1. 先确定当前目录并列出文件 ls -la
        #   2. find 命令，查找文件
        #   3. 创建文件的命令。
        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:  # 没有工具调用 => 任务结束
            return response.output_text

        for call in tool_calls:
            args = json.loads(call.arguments or "{}")
            command = args.get("command", "")
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])
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