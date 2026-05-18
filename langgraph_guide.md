# LangGraph: Stateful, Multi-Step Workflows for LLMs

LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with graph-based workflow primitives.

## Core Concepts

### StateGraph
The `StateGraph` is the fundamental building block. It manages a shared state dict that flows through nodes.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    messages: list[str]
    count: int

graph = StateGraph(MyState)
```

### Nodes
Nodes are Python functions or callables that take the current state and return an updated state.

```python
def my_node(state: dict) -> dict:
    # Process state
    return {"count": state["count"] + 1}

graph.add_node("my_node", my_node)
```

### Edges
Edges define the flow between nodes.

```python
# Fixed edge
graph.add_edge("node_a", "node_b")

# Entry and exit points
graph.set_entry_point("start_node")
graph.add_edge("final_node", END)
```

### Conditional Edges
Conditional edges route to different nodes based on the current state.

```python
def router(state: dict) -> str:
    if state["score"] > 0.8:
        return "high_quality"
    else:
        return "retry"

graph.add_conditional_edges(
    "grader",
    router,
    {
        "high_quality": "generator",
        "retry": "retriever"
    }
)
```

### Compiling and Running
```python
app = graph.compile()

# Synchronous invocation
result = app.invoke({"messages": [], "count": 0})

# Stream intermediate states
for event in app.stream(initial_state):
    print(event)
```

## Building a RAG Agent with LangGraph

### State Design for RAG
```python
class RAGState(TypedDict):
    question: str
    rewritten_question: str
    retrieved_docs: list[dict]
    relevant_docs: list[dict]
    answer: str
    retries: int
```

### Retrieval Node
```python
def retrieve(state: RAGState) -> RAGState:
    query = state["rewritten_question"] or state["question"]
    docs = vectorstore.similarity_search(query, k=5)
    return {"retrieved_docs": [{"content": d.page_content, "source": d.metadata} for d in docs]}
```

### Document Grading Node
The grader uses an LLM to assess relevance:

```python
from langchain_core.prompts import ChatPromptTemplate

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "Grade the document as 'yes' (relevant) or 'no' (not relevant)."),
    ("human", "Question: {question}\n\nDocument: {document}")
])

def grade_documents(state: RAGState) -> RAGState:
    llm = get_llm()
    relevant = []
    for doc in state["retrieved_docs"]:
        result = llm.invoke(grader_prompt.format(
            question=state["question"],
            document=doc["content"]
        ))
        if "yes" in result.content.lower():
            relevant.append(doc)
    return {"relevant_docs": relevant}
```

### Query Rewriting Node
When no relevant docs are found, rewrite the query:

```python
def rewrite_query(state: RAGState) -> RAGState:
    prompt = f"Rewrite this query to improve retrieval: {state['question']}"
    rewritten = llm.invoke(prompt).content
    return {
        "rewritten_question": rewritten,
        "retries": state["retries"] + 1
    }
```

### Routing Logic
```python
def route(state: RAGState) -> str:
    if state["relevant_docs"]:
        return "generate"
    if state["retries"] < 2:
        return "rewrite"
    return "generate"  # fallback: generate "I don't know"
```

## Checkpointing and Persistence
LangGraph supports saving graph state between turns for long-running workflows:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Use thread_id for persistent sessions
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(state, config=config)
```

## Streaming
LangGraph supports token-by-token streaming:

```python
async for chunk in app.astream_events(state, version="v1"):
    if chunk["event"] == "on_llm_stream":
        print(chunk["data"]["chunk"].content, end="", flush=True)
```

## Multi-Agent Patterns

### Supervisor Agent
A supervisor LLM decides which sub-agent to call next:

```python
def supervisor(state):
    decision = llm.invoke(f"Which agent should handle: {state['task']}?")
    return {"next_agent": decision.content}
```

### Parallel Execution
```python
from langgraph.graph import StateGraph

graph.add_node("branch_a", node_a)
graph.add_node("branch_b", node_b)
graph.add_node("merge", merge_results)

# Both branches feed into merge
graph.add_edge("branch_a", "merge")
graph.add_edge("branch_b", "merge")
```

## Error Handling in Graphs
Use try/except inside nodes and propagate errors via state:

```python
def safe_node(state: dict) -> dict:
    try:
        result = risky_operation(state)
        return {"result": result, "error": None}
    except Exception as e:
        return {"error": str(e), "result": None}
```

## Debugging with LangSmith
LangGraph integrates with LangSmith for tracing:

```bash
export LANGCHAIN_API_KEY=ls__...
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=my-rag-project
```

Every graph invocation is then visible in the LangSmith UI with full state diffs between nodes.

## Performance Tips
- Use `async` nodes (`async def`) for I/O-bound operations (LLM calls, DB queries)
- Batch LLM calls in grading nodes to reduce latency
- Cache embeddings to avoid re-computing for identical queries
- Use `.batch()` on chains instead of looping `.invoke()` calls
