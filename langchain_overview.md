# LangChain Overview

LangChain is a framework for developing applications powered by large language models (LLMs).

## Core Concepts

### Chains
Chains are sequences of calls to LLMs or other utilities. The simplest chain combines a prompt template with an LLM call. More complex chains can incorporate multiple LLMs, tools, or external data sources.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])
model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model
response = chain.invoke({"input": "What is LangChain?"})
```

### Prompts
LangChain provides a rich set of prompt templates:
- **ChatPromptTemplate**: For chat-based models with system/human/ai messages
- **PromptTemplate**: For plain text prompts with variable substitution
- **FewShotPromptTemplate**: Includes example inputs/outputs to guide the model

### Output Parsers
Output parsers transform LLM responses into structured formats:
- **StrOutputParser**: Returns the response as a plain string
- **JsonOutputParser**: Parses JSON-formatted responses
- **PydanticOutputParser**: Validates output against a Pydantic model

## LCEL (LangChain Expression Language)
LCEL is a declarative pipeline syntax using the `|` operator:

```python
chain = prompt | model | StrOutputParser()
result = chain.invoke({"question": "Explain LCEL"})
```

Key benefits:
- Streaming support out-of-the-box
- Async execution with `.ainvoke()`
- Batch processing with `.batch()`
- Composable and readable pipeline definitions

## Retrievers
Retrievers fetch documents relevant to a query from a vector store or other source.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
docs = retriever.invoke("How does RAG work?")
```

### Retrieval Modes
- `similarity`: Standard cosine similarity (default)
- `mmr`: Maximum Marginal Relevance — balances relevance and diversity
- `similarity_score_threshold`: Filters by minimum score

## Document Loaders
LangChain includes loaders for many file types:

```python
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, WebBaseLoader, CSVLoader
)

loader = PyPDFLoader("document.pdf")
docs = loader.load()
```

## Text Splitters
Split large documents into smaller chunks for embedding:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)
```

The `RecursiveCharacterTextSplitter` tries each separator in order, recursively splitting until chunks are under the size limit.

## Memory
LangChain provides conversation memory modules:

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # keep last 5 turns
```

Types of memory:
- **ConversationBufferMemory**: Full history
- **ConversationBufferWindowMemory**: Sliding window
- **ConversationSummaryMemory**: Summarised history for long conversations

## Agents
Agents use LLMs to decide which tools to call and in what order.

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = create_tool_calling_agent(model, [search], prompt)
executor = AgentExecutor(agent=agent, tools=[search])
```

## Callbacks
Callbacks hook into the chain execution lifecycle:

```python
from langchain_core.callbacks import StdOutCallbackHandler

chain.invoke({"input": "Hello"}, config={"callbacks": [StdOutCallbackHandler()]})
```

## Common Troubleshooting

### Rate Limit Errors
Use exponential backoff with `tenacity` or enable LangSmith tracing to identify bottlenecks:

```python
from langchain_core.rate_limiters import InMemoryRateLimiter
rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)
model = ChatOpenAI(rate_limiter=rate_limiter)
```

### Context Length Exceeded
Split documents more aggressively or use a model with a larger context window. Consider using a `MapReduceDocumentsChain` for summarisation over large corpora.

### Serialization Issues
When saving/loading chains, use `chain.save("chain.yaml")` (not all chain types support this). For custom components, ensure they implement `_chain_type` properly.
