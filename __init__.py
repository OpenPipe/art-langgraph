"""LangGraph Training - Training framework for LangGraph agents."""

__version__ = "0.1.0"

from .framework import TrainingFramework, TrainingConfig, train
from .judge import JudgeGroupScore, JudgeGroupResponse, get_judge_completion_msg
from .logging import FileLogger
from .llm_wrapper import LoggingLLM, init_chat_model, add_thread
from .message_utils import (
    langchain_msg_to_openai,
    convert_langgraph_messages,
    make_message_param,
    Message,
    MessagesAndChoices
)

__all__ = [
    # Framework components
    "TrainingFramework",
    "TrainingConfig", 
    "train",
    
    # Judge/scoring
    "JudgeGroupScore",
    "JudgeGroupResponse",
    "get_judge_completion_msg",
    
    # Logging
    "FileLogger",
    
    # LLM wrapper
    "LoggingLLM",
    "init_chat_model",
    "add_thread",
    
    # Message utilities
    "langchain_msg_to_openai",
    "convert_langgraph_messages", 
    "make_message_param",
    "Message",
    "MessagesAndChoices"
]