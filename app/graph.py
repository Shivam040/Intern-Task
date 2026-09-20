from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.rag import generate_grounded_answer, retrieve


class RAGState(TypedDict, total=False):
    question: str
    context: list[dict]
    score: float
    is_relevant: bool
    answer: str


def retrieve_context(state: RAGState) -> RAGState:
    context = retrieve(state["question"])
    top_score = context[0]["score"] if context else 0.0
    return {"context": context, "score": top_score}


def check_relevance(state: RAGState) -> RAGState:
    is_relevant = bool(state.get("context")) and state.get("score", 0.0) >= settings.relevance_threshold
    return {"is_relevant": is_relevant}


def route_after_check(state: RAGState) -> Literal["generate_answer", "fallback"]:
    return "generate_answer" if state.get("is_relevant", False) else "fallback"


def generate_answer(state: RAGState) -> RAGState:
    answer = generate_grounded_answer(state["question"], state.get("context", []))
    return {"answer": answer}


def fallback(state: RAGState) -> RAGState:
    return {"answer": "I could not find sufficient information in the provided knowledge base."}


def build_graph():
    builder = StateGraph(RAGState)

    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("check_relevance", check_relevance)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("fallback", fallback)

    builder.add_edge(START, "retrieve_context")
    builder.add_edge("retrieve_context", "check_relevance")
    builder.add_conditional_edges(
        "check_relevance",
        route_after_check,
        {
            "generate_answer": "generate_answer",
            "fallback": "fallback",
        },
    )
    builder.add_edge("generate_answer", END)
    builder.add_edge("fallback", END)

    return builder.compile()


rag_graph = build_graph()
