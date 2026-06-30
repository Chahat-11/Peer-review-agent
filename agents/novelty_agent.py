import sys
sys.path.insert(0, '/Users/chahatsingh/Desktop/peer-review-agent')
import json
from groq import Groq
from pipeline.retrieval import PaperRetriever
from dotenv import load_dotenv
import os

load_dotenv()

NOVELTY_SYSTEM_PROMPT = """You are a specialist reviewer whose only job is to assess the novelty and originality of a research paper. You will be given the paper text and 5 similar papers retrieved from recent literature.

Evaluate:
1. Does the paper introduce a new method, architecture, or technique?
2. Is there a clear gap between this paper and existing work?
3. Are the claims of novelty well-supported?

Output ONLY valid JSON with these keys:
- novelty_score: integer from 1 to 10 (1=not novel at all, 10=highly novel)
- key_claims: list of strings (the main novel claims)
- gap_assessment: string (how this paper differs from prior work)
- similar_papers_considered: list of strings (brief descriptions of the similar papers)
- confidence: string (low/medium/high)
"""

def run_novelty_agent(paper_dict, similar_papers, model="llama-3.3-70b-versatile"):
    """Run the novelty agent on a paper."""
    client = Groq()

    # Prepare the input
    paper_text = paper_dict.get("abstract", "") or paper_dict.get("full_text", "")[:3000]

    similar_text = ""
    for i, sp in enumerate(similar_papers):
        similar_text += f"\nSimilar Paper {i+1}: {sp['abstract'][:300]}\n"

    user_message = f"""Paper to evaluate:
{paper_text}

Similar papers from recent literature:
{similar_text}

Provide your evaluation as JSON."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": NOVELTY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
    )

    result = json.loads(response.choices[0].message.content)
    return result

if __name__ == "__main__":
    # Quick test
    from pipeline.pdf_ingestion import extract_paper

    import sys
    if len(sys.argv) < 2:
        print("Usage: python novelty_agent.py <pdf_path>")
        sys.exit(1)

    paper = extract_paper(sys.argv[1])
    print(f"Paper loaded: {len(paper['full_text'])} chars")

    # For testing, use dummy similar papers
    dummy_similar = [{"abstract": "This paper proposes a graph neural network for molecular property prediction."}] * 5

    result = run_novelty_agent(paper, dummy_similar)
    print("\n=== Novelty Agent Result ===")
    print(json.dumps(result, indent=2))