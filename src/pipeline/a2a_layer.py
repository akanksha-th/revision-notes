from typing import TypedDict, List, Dict, Any
from datetime import datetime
import asyncio


class AgentMessage(TypedDict):
    from_agent: str
    to_agent: str
    timestamp: datetime
    payload: Dict[str, Any]
    message_type: str   # "request", "response", "broadcast"


class SharedMessageBus(TypedDict):
    """Central communication hub for agents"""
    messages: List[AgentMessage]
    subscriptions: Dict[str, List[str]]     # agent_id –> [message_types]