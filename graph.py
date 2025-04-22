from typing_extensions import TypedDict
from typing import Optional, List, Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from models import *
from core import ReasoningNodes, ModelHandler

class ToolResponse(TypedDict):
    tool_name: str
    tool_response: str

class State(TypedDict):
    user_query: str
    model_name: str
    cot_breakdown: List[str]
    tools_needed: List[str]
    tools_reslts: List[ToolResponse]
    answer: str
    score: int
    feedback: str
    model_nodes: ModelHandler
    reasoning_nodes: ReasoningNodes
    status: Literal["initializing", "running", "completed", "failed", "stopped"]

#Nodes

def model_initialization(state: State) -> State:
    """
    Initialize the model.
    """
    model_nodes = ModelHandler(model_name=state["model_name"])
    model_nodes._load_model()
    reasoning_nodes = ReasoningNodes()
    state["model_name"] = model_nodes.model_name
    state["model_nodes"] = model_nodes
    state["reasoning_nodes"] = reasoning_nodes
    state["status"] = "initializing"
    return state

def model_unloading(state: State) -> State:
    state["model_nodes"]._unload_model()
    state["status"] = "stopped"
    state["model_nodes"] = None
    state["reasoning_nodes"] = None
    return state

def CoT_Gen(state: State) -> State:
    """
    Generate Chain of Thought (CoT) using the model.
    """
    state["status"] = "running"
    model_nodes = state["model_nodes"]
    reasoning_nodes = state["reasoning_nodes"]
    response = reasoning_nodes.CoT_generate(state["user_query"], model_nodes)
    state["cot_breakdown"] = response["modular_breakdown"]
    
    


workflow = StateGraph(State)
workflow.add_node("Model Initialization", model_initialization)

def cot_generation()
    