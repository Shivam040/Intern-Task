from app import graph

def test_relevant_context_routes_to_generation(monkeypatch):
    monkeypatch.setattr(graph.settings, "relevance_threshold", 0.60)
    state = {"context": [{"text": "x", "page": 1, "score": 0.72}], "score": 0.72}

    update = graph.check_relevance(state)
    routed_state = {**state, **update}

    assert update["is_relevant"] is True
    assert graph.route_after_check(routed_state) == "generate_answer"


def test_low_score_routes_to_fallback(monkeypatch):
    monkeypatch.setattr(graph.settings, "relevance_threshold", 0.60)
    state = {"context": [{"text": "x", "page": 1, "score": 0.12}], "score": 0.12}

    update = graph.check_relevance(state)
    routed_state = {**state, **update}

    assert update["is_relevant"] is False
    assert graph.route_after_check(routed_state) == "fallback"


def test_fallback_message_is_grounded_failure_message():
    result = graph.fallback({"question": "Who won the World Cup?"})
    assert result["answer"] == "I could not find sufficient information in the provided knowledge base."
