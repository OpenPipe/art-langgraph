"""LLM wrapper with logging functionality."""

import uuid
import contextvars
import os
from typing import Literal, Any
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from .logging import FileLogger

CURRENT_CONFIG = contextvars.ContextVar("CURRENT_CONFIG")

mappings = {}

def add_thread(thread_id, base_url, api_key, model):
    os.makedirs(os.path.dirname('.art/langgraph/{thread_id}'), exist_ok=True)
    CURRENT_CONFIG.set({
        "logger": FileLogger(f'.art/langgraph/{thread_id}'),
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    })

def init_chat_model(
    model: str,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal[None] = None,
    config_prefix: str | None = None,
    **kwargs: Any,
    ):
    config = CURRENT_CONFIG.get()
    return LoggingLLM(
        ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["model"],
            temperature=1.0,
        ),
        config["logger"]
    )

class LoggingLLM(Runnable):
    def __init__(self, llm, logger):
        self.llm = llm
        self.logger = logger

    def _log(self, completion_id, input, output):
        if self.logger:
            entry = {"input": input, "output": output}
            self.logger.log(f"{completion_id}", entry)

    def invoke(self, input, config=None):
        completion_id = str(uuid.uuid4())

        result = self.llm.invoke(input, config=config)
        self._log(completion_id, input, result)

        if hasattr(result, 'get') and result.get('parsed'):
            return result.get('parsed')
        return result

    async def ainvoke(self, input, config=None):
        completion_id = str(uuid.uuid4())

        result = await self.llm.ainvoke(input, config=config)
        self._log(completion_id, input, result)

        if hasattr(result, 'get') and result.get('parsed'):
            return result.get('parsed')
        return result

    def with_structured_output(self, tools):
        return LoggingLLM(self.llm.with_structured_output(tools, include_raw=True), self.logger)

    def bind_tools(self, tools):
        return LoggingLLM(self.llm.bind_tools(tools), self.logger)

    # def __getattr__(self, attr):
    #     return getattr(self.llm, attr)