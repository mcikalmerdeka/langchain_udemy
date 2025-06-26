from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Union, List, Dict, Optional
from uuid import UUID

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running"""
        print(f"\n***Prompt to LLM was:***\n{prompts[0]}")
        print(f"******************************************")
    
    def on_llm_end(
        self, response: LLMResult, **kwargs: Any
    ) -> Any:
        """Run when LLM ends running"""
        print(f"***LLM response was:***\n{response.generations[0][0].text}")
        print(f"******************************************")