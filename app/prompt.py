def build_prompt(question, retrieved_chunks):
    context_parts = []

    for retrieved_chunk in retrieved_chunks:

    # create text such as:
    #
    # [Source: gpu_infrastructure.md]
    # <chunk content>
    #
    # and add it to context_parts
        context_parts.append(f"[Source: {retrieved_chunk['source']} \n {retrieved_chunk['content']}]")

    # Combine context_parts into one string
    combined = '\n\n'.join(context_parts)


    # Create the final prompt containing:
    #
    # instructions
    # context
    # question
    prompt = f'''
You are an internal assistant for Northstar Technologies.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer the question,
say that the available documentation does not contain enough information.

Do not use outside knowledge.

Cite the source document names used.

Context: {combined}

Question: {question}'''


    return prompt