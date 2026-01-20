from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Optional, Any, List
import pandas as pd
import asyncio
from datetime import datetime
from src.pipeline.a2a_layer import AgentMessage


class DataUnderstandingState(TypedDict):
    processed_dataset_path: str
    target_column: str
    column_types: Dict[str, str]
    _df: Optional[Any]

    _message_bus: List[AgentMessage]

    dataset_summary: Optional[Dict]
    missingness: Optional[Dict]
    cardinality: Optional[Dict]
    target_analysis: Optional[Dict]
    eda_artifacts: Optional[Dict]

# ----------
# Nodes
# ----------

async def load_artifact(state: DataUnderstandingState) -> DataUnderstandingState:
    state["_df"] = pd.read_parquet(state["processed_dataset_path"])
    return state


async def stats_node(state: DataUnderstandingState) -> DataUnderstandingState:
    df = state["_df"]
    await asyncio.sleep(1)
    state["dataset_summary"] = {
        "rows": len(df),
        "columns": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2
    }

    state["missingness"] = df.isnull().sum().to_dict()
    state["cardinality"] = df.nunique().to_dict()
    return state


async def eda_node(state: DataUnderstandingState) -> DataUnderstandingState:
    message = AgentMessage(
        from_agent="eda_agent",
        to_agent="feature_engineering_agent",
        timestamp=datetime.now(),
        payload={"high_correlation_pairs": [...], "skewed_features": [...]},
        message_type="broadcast"
    )
    state["_message_bus"].append(message)
    return state


async def target_node(state: DataUnderstandingState) -> DataUnderstandingState:
    df = state["_df"]
    target = state["target_column"]
    state["target_analysis"] = {
        "type": state["column_types"][target],
        "distribution": df[target].value_counts().to_dict() if state["column_types"][target] == "categorical" else df[target].describe().to_dict()
    }
    return state


async def merge_node(state: DataUnderstandingState) -> DataUnderstandingState:
    state["_df"] = None
    return state

# ----------
# Graph
# ----------

def build_data_understanding_graph():
    graph = StateGraph(DataUnderstandingState)

    graph.add_node("load", load_artifact)
    graph.add_node("stats", stats_node)
    graph.add_node("eda", eda_node)
    graph.add_node("target", target_node)
    graph.add_node("merge", merge_node)

    graph.set_entry_point("load")

    graph.add_edge("load", "stats")
    graph.add_edge("load", "eda")
    graph.add_edge("load", "target")

    graph.add_edge("stats", "merge")
    graph.add_edge("eda", "merge")
    graph.add_edge("target", "merge")
    
    graph.add_edge("merge", END)

    return graph.compile()


if __name__ == "__main__":
    understanding_graph = build_data_understanding_graph()

    result = understanding_graph.ainvoke({})
    print(result)