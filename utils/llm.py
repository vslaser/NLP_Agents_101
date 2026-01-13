from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from utils.settings import load_settings, Settings
from utils.exceptions import ExternalServiceError, ConfigError
from utils.logger import get_logger
from utils.messages import ChatMessage
from utils.prompt_loader import load_prompt
from utils.validation import parse_json_strict

log = get_logger(__name__)

# ---------- Simple LLM client (for chatbot + RAG) ----------

class LLMClient:
    def chat(self, messages: list[ChatMessage]) -> str:
        raise NotImplementedError

@dataclass
class RuntimeLLMConfig:
    temperature: float | None = None
    max_new_tokens: int | None = None

class LocalTransformersCPU(LLMClient):
    def __init__(self, settings: Settings, rt: RuntimeLLMConfig | None = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.settings = settings
        self.rt = rt or RuntimeLLMConfig()

        self.model_id = settings.local_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _format(self, messages: list[ChatMessage]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [m.model_dump(exclude_none=True) for m in messages if m.role != "tool"],
                tokenize=False,
                add_generation_prompt=True,
            )
        lines = []
        for m in messages:
            if m.role == "tool":
                lines.append(f"TOOL({m.name}): {m.content}")
            else:
                lines.append(f"{m.role.upper()}: {m.content}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def chat(self, messages: list[ChatMessage]) -> str:
        import torch

        prompt = self._format(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        temp = self.rt.temperature if self.rt.temperature is not None else self.settings.local_temperature
        mx = self.rt.max_new_tokens if self.rt.max_new_tokens is not None else self.settings.local_max_new_tokens

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(mx),
                do_sample=temp > 0,
                temperature=float(temp) if temp > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):].strip()

class OpenAICompatAPI(LLMClient):
    def __init__(self, settings: Settings):
        import httpx
        self.httpx = httpx
        self.settings = settings
        if not settings.openai_compat_api_key or not settings.openai_compat_base_url:
            raise ConfigError("API mode selected but OPENAI_COMPAT_* is not configured.")
        self.base_url = settings.openai_compat_base_url.rstrip("/")
        self.api_key = settings.openai_compat_api_key
        self.model = settings.openai_compat_model

    def chat(self, messages: list[ChatMessage]) -> str:
        try:
            r = self.httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": [m.model_dump(exclude_none=True) for m in messages if m.role != "tool"]},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise ExternalServiceError(str(e))

def get_llm(runtime: RuntimeLLMConfig | None = None) -> LLMClient:
    s = load_settings()
    if s.llm_mode == "api":
        return OpenAICompatAPI(s)
    return LocalTransformersCPU(s, runtime)

# ---------- LangChain ChatModel adapter (for agents + LangGraph) ----------

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

class LocalCpuToolCallingChatModel:
    """
    Local CPU model that *simulates* tool calling:
    - We inject tool schema and enforce JSON tool-call output.
    - We parse JSON and convert to AIMessage(tool_calls=...).
    """

    def __init__(self, settings: Settings, runtime: RuntimeLLMConfig | None = None):
        self.settings = settings
        self.runtime = runtime or RuntimeLLMConfig()
        self._llm = LocalTransformersCPU(settings, self.runtime)
        self._tools: list[BaseTool] = []

    def bind_tools(self, tools: list[BaseTool]) -> "LocalCpuToolCallingChatModel":
        clone = LocalCpuToolCallingChatModel(self.settings, self.runtime)
        clone._tools = tools
        return clone

    def _tools_spec(self) -> str:
        """Generate a simple, concise spec of available tools with usage examples."""
        lines = []
        for t in self._tools:
            # Include name and description
            lines.append(f"- {t.name}: {t.description}")
        return "\n".join(lines)

    def _validate_tool_call(self, tool_name: str, args: dict) -> bool:
        """Check if the tool name exists and has valid argument structure."""
        for t in self._tools:
            if t.name == tool_name:
                # Tool exists; for now, just ensure args is a dict
                return True
        return False

    def _detect_tool_from_text(self, user_text: str) -> dict | None:
        """Fallback: detect tool from natural language keywords in user text."""
        text_lower = user_text.lower()
        
        # Check for add/plus
        if any(word in text_lower for word in ["add", "plus", "+", "sum"]):
            # Try to extract two numbers
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', user_text)
            if len(numbers) >= 2:
                return {"tool": "add", "args": {"a": float(numbers[0]), "b": float(numbers[1])}}
        
        # Check for multiply/times
        if any(word in text_lower for word in ["multiply", "times", "multiply", "x", "*"]):
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', user_text)
            if len(numbers) >= 2:
                return {"tool": "multiply", "args": {"a": float(numbers[0]), "b": float(numbers[1])}}
        
        # Check for create note
        if any(word in text_lower for word in ["create note", "save note", "new note"]):
            return {"tool": "create_note", "args": {"title": "note", "body": user_text}}
        
        # Check for list notes
        if any(word in text_lower for word in ["list notes", "show notes", "my notes"]):
            return {"tool": "list_notes", "args": {"limit": 10}}
        
        # Default to none
        return {"tool": "none", "args": {}}

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        from langchain_core.messages import ToolMessage
        
        # Check if there are recent tool results in the messages
        has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
        
        # Find the most recent human message
        user = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user = m.content
                break

        # If we have tool results, the agent should synthesize an answer
        if has_tool_results:
            # Use the regular agent prompt to provide a final answer
            answer = self._llm.chat([
                ChatMessage(role="system", content=load_prompt("agent_single_system.txt")),
                ChatMessage(role="user", content=user),
            ])
            return AIMessage(content=answer, tool_calls=[])

        system = load_prompt("tool_call_json_system.txt")
        # Add tool specs directly to the system prompt
        tools_spec = self._tools_spec()
        system = system + "\n\nAvailable tools:\n" + tools_spec

        raw = self._llm.chat([
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ])

        # Parse and produce tool_calls
        # Try to extract JSON from the response (in case model adds extra text)
        try:
            # First try direct parsing
            obj = parse_json_strict(raw)
        except Exception as e1:
            # Try to find JSON object in the response using improved extraction
            import re
            # Look for { ... } patterns, handling nested braces
            start_idx = raw.find('{')
            if start_idx != -1:
                # Find matching closing brace
                depth = 0
                end_idx = -1
                for i in range(start_idx, len(raw)):
                    if raw[i] == '{':
                        depth += 1
                    elif raw[i] == '}':
                        depth -= 1
                        if depth == 0:
                            end_idx = i + 1
                            break
                
                if end_idx != -1:
                    json_str = raw[start_idx:end_idx]
                    try:
                        obj = parse_json_strict(json_str)
                    except Exception as e2:
                        # JSON extraction failed, use natural language fallback
                        log.debug(f"JSON extraction failed, using natural language tool detection")
                        obj = self._detect_tool_from_text(user)
                        if not obj:
                            log.warning(f"Failed to parse or detect tool from model response: {raw[:200]}...")
                            return AIMessage(content="I couldn't understand which tool to use. Please clarify your request.", tool_calls=[])
                else:
                    # No closing brace found, use natural language fallback
                    obj = self._detect_tool_from_text(user)
                    if not obj:
                        log.warning(f"Failed to parse or detect tool from model response: {raw[:200]}...")
                        return AIMessage(content="I couldn't understand which tool to use. Please clarify your request.", tool_calls=[])
            else:
                # No JSON found at all, use natural language fallback
                obj = self._detect_tool_from_text(user)
                if not obj:
                    log.warning(f"No JSON or tool keywords detected in response")
                    return AIMessage(content="I couldn't understand which tool to use. Please clarify your request.", tool_calls=[])

        tool = obj.get("tool", "none")
        args = obj.get("args", {})
        if tool == "none":
            # For local mode, provide a second pass to answer normally (non-tool)
            answer = self._llm.chat([
                ChatMessage(role="system", content=load_prompt("agent_single_system.txt")),
                ChatMessage(role="user", content=user),
            ])
            return AIMessage(content=answer, tool_calls=[])

        if not isinstance(args, dict):
            args = {}

        # Validate that the tool exists
        if not self._validate_tool_call(tool, args):
            log.warning(f"Model attempted to call non-existent tool: {tool}. Available tools: {[t.name for t in self._tools]}")
            return AIMessage(content=f"Tool '{tool}' not found. Please use one of: {', '.join(t.name for t in self._tools)}", tool_calls=[])

        return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": "call_1", "type": "tool_call"}])

def get_chat_model_for_agents(tools: list[BaseTool], runtime: RuntimeLLMConfig | None = None):
    s = load_settings()
    if s.llm_mode == "api":
        # Native tool calling
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
            model=s.openai_compat_model,
            api_key=s.openai_compat_api_key,
            base_url=s.openai_compat_base_url,
            temperature=0,
        )
        return model.bind_tools(tools)
    # Local CPU tool calling adapter
    return LocalCpuToolCallingChatModel(s, runtime).bind_tools(tools)
