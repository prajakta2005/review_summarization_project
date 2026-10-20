"""
llm_explainer.py
----------------
Generates a natural-language explanation of a review's credibility
assessment using Google's Gemini API, as a companion to
explainability.py's rule-based explanation.

Design choice -- read before wiring this in:
Gemini is used as a NARRATOR over facts we already computed, not as a
second, independent credibility judge. The prompt hands it the review
text, the score, and the already-computed positive/negative factors
from explainability.py, and explicitly tells it not to invent new
claims. This keeps the actual credibility decision fully auditable
(it still comes from the RandomForest + rule-based factor extraction)
while letting Gemini write a more natural sentence for the UI.

Setup:
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"
    (get a free key at https://aistudio.google.com/apikey)

Uses the current `google-genai` package -- NOT the older
`google-generativeai`, which Google has deprecated.
"""

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Create a file named .env in the "
            "project root (same folder as app.py) containing a line: "
            "GEMINI_API_KEY=your-key-here. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    _client = genai.Client(api_key=api_key)
    return _client


def generate_llm_explanation(review_row, positive_factors, negative_factors, model: str = "gemini-2.0-flash") -> str:
    """review_row: dict/Series with review_text, credibility_score, credibility_label.
    positive_factors / negative_factors: lists of plain-English phrases,
    already computed by explainability.explain_review() -- passed in so
    Gemini narrates REAL computed factors, never invents its own.
    """
    client = _get_client()

    prompt = f"""You are explaining an automated review-credibility assessment to a
non-technical user. Do NOT invent any new facts, numbers, or claims about the
review beyond what is given below -- only explain and connect the given facts
in clear, friendly language. Keep it to 2-3 sentences.

Review text: "{review_row['review_text']}"
Credibility score: {review_row['credibility_score']}% ({review_row['credibility_label']})
Positive factors (support credibility): {positive_factors if positive_factors else "none"}
Negative factors (raise concern): {negative_factors if negative_factors else "none"}

Write a short, natural-language explanation of why this review received this
credibility assessment, using only the factors listed above."""

    response = client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()


def generate_llm_explanation_safe(review_row, positive_factors, negative_factors, fallback_text: str) -> str:
    """Never crashes the app -- falls back to the rule-based explanation
    text if the API key is missing, the network call fails, or Gemini
    returns something unusable. Call this one from app.py.
    """
    try:
        return generate_llm_explanation(review_row, positive_factors, negative_factors)
    except Exception as e:
        return f"{fallback_text}\n\n_(Gemini explanation unavailable: {e})_"