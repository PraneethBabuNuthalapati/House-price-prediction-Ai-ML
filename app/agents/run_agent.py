from app.agents.langgraph_agent import build_graph

graph = build_graph()

def run_agent(req, query: str):
    result = graph.invoke({
        "req": req,
        "query": query
    })
    
    response = {
        "final_price": result["prediction"]["final_price"]
    }
    
    explanation_block = result.get("explanation")
    
    if explanation_block:
        response["top_factors"] = explanation_block.get("top_factors", [])
        response["explanation"] = explanation_block.get("explanation", "")
        
    print("Agent response:", response)
    return response
