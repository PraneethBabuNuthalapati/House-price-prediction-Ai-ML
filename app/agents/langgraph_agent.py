from typing import Any, Literal, TypedDict
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.agents.agent_tools import predict_price_tool, explain_tool

load_dotenv()


class AgentState(TypedDict, total=False):
    req: Any
    query: str
    mode: str
    prediction: dict[str, Any]
    explanation: dict[str, Any]


llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    temperature=0
)


def decide_node(state: AgentState):
    query = str(state.get("query", "predict price"))

    prompt = f"""
You are an intent router for a house price AI system.

Decide whether the user wants:
- PRICE -> only the predicted value
- FULL -> prediction plus explanation/breakdown/reasoning

Examples:
- "give me the price" -> PRICE
- "predict the house value" -> PRICE
- "explain the price" -> FULL
- "breakdown the price" -> FULL
- "why is this expensive?" -> FULL
- "what factors affect the value?" -> FULL

Return only one word: PRICE or FULL

User query: {query}
"""

    try:
        decision = llm.invoke(prompt).content.strip().upper()
        state["mode"] = "full" if "FULL" in decision else "price"
    except Exception:
        state["mode"] = "price"

    return state


def predict_node(state: AgentState):
    req = state.get("req")
    if req is None:
        raise ValueError("Missing 'req' in agent state")

    state["prediction"] = predict_price_tool(req)
    return state


def explain_node(state: AgentState):
    prediction = state.get("prediction")
    if prediction is None or "df" not in prediction:
        raise ValueError("Missing prediction dataframe for explanation")

    state["explanation"] = explain_tool(prediction["df"])
    return state


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("decide", decide_node)
    builder.add_node("predict", predict_node)
    builder.add_node("explain", explain_node)

    builder.set_entry_point("decide")
    builder.add_edge("decide", "predict")

    def route(state: AgentState) -> Literal["explain", "__end__"]:
        return "explain" if state["mode"] == "full" else END

    builder.add_conditional_edges("predict", route)
    builder.add_edge("explain", END)

    return builder.compile()