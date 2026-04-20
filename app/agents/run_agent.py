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
    
    if "explaination" in result:
        response["top_factors"] = result["explaination"].get("top_factors", [])
        response["explaination"] = result["explaination"].get("explaination", "")
        
    return response
