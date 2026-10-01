"""Question heuristics shared by the one-shot pipeline and the agent."""


def estimate_question_complexity(question: str) -> int:
    """
    Estimate question complexity to determine how many articles to retrieve

    Args:
        question: User's question

    Returns:
        Number of articles to retrieve (3-7)
    """
    question_lower = question.lower()

    # Complex question indicators
    complexity_score = 0

    # Multi-part questions (need multiple perspectives)
    if ' and ' in question_lower:
        complexity_score += 2
    if ' vs ' in question_lower or ' versus ' in question_lower:
        complexity_score += 3  # Comparisons need both sides

    # Comparison/relationship questions (need context from multiple articles)
    if any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs']):
        complexity_score += 3
    if any(word in question_lower for word in ['relationship', 'connect', 'relate', 'impact', 'affect', 'influence', 'cause']):
        complexity_score += 2

    # Deep/analytical questions (need comprehensive context)
    if any(word in question_lower for word in ['how does', 'how do', 'why', 'explain']):
        complexity_score += 2
    if any(word in question_lower for word in ['history', 'evolution', 'development', 'origin']):
        complexity_score += 2

    # Broad conceptual questions
    if any(word in question_lower for word in ['overview', 'summary', 'introduction', 'basics']):
        complexity_score += 1

    # Future/prediction questions (need current state + theories)
    if any(word in question_lower for word in ['future', 'prediction', 'will', 'going to']):
        complexity_score += 2

    # Long questions often need more context
    if len(question.split()) > 12:
        complexity_score += 1

    # Map complexity to number of articles
    # With better selection AI, we can retrieve more targeted articles
    if complexity_score >= 6:
        return 6  # Very complex - retrieve 6 articles
    elif complexity_score >= 4:
        return 5  # Complex - retrieve 5 articles
    elif complexity_score >= 3:
        return 4  # Moderate-complex - retrieve 4 articles
    elif complexity_score >= 2:
        return 4  # Moderate - retrieve 4 articles
    else:
        return 3  # Simple - retrieve 3 articles (minimum)


def passage_budget(question: str) -> int:
    """Passages to retrieve for a question: 3-6 "articles" of complexity map to 8-12 passages"""
    return min(12, 2 * estimate_question_complexity(question) + 2)
