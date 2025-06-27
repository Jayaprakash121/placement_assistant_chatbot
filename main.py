def get_rag_chain(model_name="gemini-2.0-flash", api_key=None):
    from create_db import create_or_load_chroma_db
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    db = create_or_load_chroma_db()

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    # Contextualize question prompt
    # This system prompt helps the AI understand that it should reformulate the question
    # based on the chat history to make it a standalone question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "Rephrase the question using prior conversation context so that it "
        "can be understood independently. For example, if the user says 'What about CGPA?', "
        "and the earlier message refers to 'internships', rewrite as 'What is the CGPA required for internships?'."
        " Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever
    # This uses the LLM to help reformulate the question based on chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer question prompt
    # This system prompt helps the AI understand that it should provide concise answers
    # based on the retrieved context and indicates what to do if the answer is unknown
    qa_system_prompt = (
        "You are a helpful, friendly, and smart assistant designed to answer placement-related queries.\n\n"
        "You must answer user questions using the retrieved documents and respond in a concise, clear, and helpful way.\n\n"
        "Relevant Documents:\n{context}\n\n"
        "Instructions:\n"
        "- If the query is a greeting like 'hi', 'hello', or 'how are you', respond warmly and do not provide any other information.\n"
        "- If the query asks about preparation for a company, do the following:\n"
        "    • First, extract and summarize the preparation guide from the documents — including skills required, interview process, eligibility, and any other relevant info.\n"
        "    • Then, check if the documents include any students placed in that company.\n"
        "        - If found, list the student names and generate their email in this format: <RollNumber>@iitbbs.ac.in\n"
        "        - Encourage the user to contact them for guidance.\n"
        "        - Example: 'You can contact [Student Name] at [Roll]@iitbbs.ac.in for preparation help.'\n"
        "        - If no students are found, simply return the preparation guide based on the documents.\n"
        "- For general placement questions (like CTC, cutoff, role), use only the retrieved context to answer accurately.\n"
        "- If no relevant info is found, just say something like this which means you are not sure about that based on the documents.\n"
        "- For casual chats, feel free to reply warmly and ask a follow-up question to keep the conversation going.\n"
    )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a chain to combine documents for question answering
    # `create_stuff_documents_chain` feeds all retrieved context into the LLM
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
