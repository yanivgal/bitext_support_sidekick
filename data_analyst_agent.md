Your task is to build a simple application backed by an agent that can answer the user's question about the Bitext - Customer Service Tagged Training dataset.

The questions/requests can be split to three main categories:
1. Structured:
    * What are the most frequent categories?
    * Show examples of Category X.
    * What categories exist?
    * Show intent distributions.
2. Unstructured:
    * Summarize Category X.
    * Summarize how agent respond to Intent Y.
3. Out-of-scope:
    * Who is Magnus Carlson?
    * What is Serj's rating?

Your task:
1. Create an architecture specification/diagram.
2. Write a simple Q&A Streamlit chat app that allows the user to ask the agent questions about the dataset.
3. [Bonus] Support follow-up questions.
4. [Bonus] Toggle between planning modes: pre-planning + execution vs. ReActive dynamic planning.

Notes:
 * You can use draw.io for the specs/diagram
 * The diagram should contain the flow as well as the tools available to the agent
 * Both Nebius AI Studio and OpenAI models can be used (Qwen2.5-32B-Instruct is a good default)
 * The agent should at least answer the question above correctly
 * Use JSON schemas where applicable
 * Think carefully which tools you want to expose to your agent for the task
 * Use the code snippets from the presentation but feel fr
 * Although your task is not evaluated on code quality but we advise you to consider the poor person who has to grade your assignment