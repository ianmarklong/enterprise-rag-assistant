def build_prompt(question, retrieved_chunks):
    context_parts = []

    for result in retrieved_chunks:
        context_parts.append(
            f"[Source: {result['source']}]\n"
            f"{result['content']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an internal assistant for Northstar Technologies.

Answer the user's question using only the provided context.

If the provided context contains enough information:
- Answer the question directly.
- Cite the source document name(s) used.
- Do NOT use the refusal phrase below anywhere in the answer.

If the provided context does NOT contain enough information to answer the question, respond with exactly:

INSUFFICIENT_DOCUMENTATION

Do not invent company policies, procedures, systems, or facts.

CONTEXT:
{context}

QUESTION:
{question}
"""

    return prompt