from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, List, Any
import pandas as pd
import os

class IngestionState(TypedDict):
    dataset_path: str
    target_column: Optional[str] = None

    dataset_loaded: bool
    num_rows: Optional[int]
    num_columns: Optional[int]
    column_types: Optional[Dict[str, str]]
    issues: Optional[List]

    processed_dataset_path: Optional[str]

    _df: Optional[Any]

# ----------
# Nodes
# ----------

def load_csv(state: IngestionState) -> IngestionState:
    path = state['dataset_path']
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at path: {path}")
    
    df = pd.read_csv(path)

    state['dataset_loaded'] = True
    state['num_rows'] = df.shape[0]
    state['num_columns'] = df.shape[1]

    state["_df"] = df   #temporary storage
    return state


def validate_dataset(state: IngestionState) -> IngestionState:
    df = state["_df"]
    issues = []

    if df.empty:
        issues.append("Dataset is empty.")

    if state["target_column"]:
        if state["target_column"] not in df.columns:
            issues.append(f"Target column '{state['target_column']}' not found in dataset.")

    if df.columns.duplicated().any():
        issues.append("Dataset contains duplicate columns.")

    state['issues'] = issues
    return state


def infer_column_types(state: IngestionState) -> IngestionState:
    df = state["_df"]
    column_types = {}

    for col in df.columns:
        if col == state.get("target_column"):
            column_types[col] = "target"
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"

    state['column_types'] = column_types
    return state


def persist_artifact(state: IngestionState) -> IngestionState:
    df = state["_df"]

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/processed_dataset.parquet"
    df.to_parquet(output_path)

    state["processed_dataset_path"] = output_path

    state["_df"] = None
    return state


# ----------
# Graph
# ----------

def build_ingestion_graph():
    graph = StateGraph(IngestionState)

    graph.add_node("load_csv", load_csv)
    graph.add_node("validate_dataset", validate_dataset)
    graph.add_node("infer_column_types", infer_column_types)
    graph.add_node("persist_artifact", persist_artifact)

    graph.set_entry_point("load_csv")

    graph.add_edge("load_csv", "validate_dataset")
    graph.add_edge("validate_dataset", "infer_column_types")
    graph.add_edge("infer_column_types", "persist_artifact")
    graph.add_edge("persist_artifact", END)

    return graph.compile()


if __name__ == "__main__":
    ingestion_graph = build_ingestion_graph()

    result = ingestion_graph.invoke({
        "dataset_path": "./data/raw/Clean_Dataset.csv",
        "target_column": "price",
        "dataset_loaded": False,
        "num_rows": None,
        "num_columns": None,
        "column_types": None,
        "issues": None,
        "processed_dataset_path": None
    })

    print(result)
