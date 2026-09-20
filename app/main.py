from fastapi import FastAPI, HTTPException
import traceback
from app.graph import rag_graph
from app.schemas import ChatRequest, ChatResponse, ContextChunk


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot grounded only in the provided Agentic AI ebook.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = rag_graph.invoke({"question": request.question.strip()})
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()

        raise HTTPException( status_code=500, detail=f"{type(exc).__name__}: {str(exc)}" ) from exc

    return ChatResponse(
        answer=result.get("answer", ""),
        context=[ContextChunk(**item) for item in result.get("context", [])],
        score=round(float(result.get("score", 0.0)), 4),
    )
