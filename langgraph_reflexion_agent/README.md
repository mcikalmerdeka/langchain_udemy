# LangGraph Reflexion Agent

## Overview
This folder implements a **Reflexion Agent** - an advanced research assistant that uses structured reflection, web search, and iterative revision to produce high-quality, well-researched answers with proper citations.

## What It Does
The Reflexion Agent creates comprehensive research responses by:
1. **Initial Research**: Generates a detailed ~250-word answer with self-critique
2. **Web Search**: Executes targeted search queries based on identified gaps
3. **Iterative Revision**: Improves the answer using new information and citations
4. **Quality Control**: Limits iterations to prevent infinite loops

## Architecture
Uses a **MessageGraph** with three main nodes:
- `FIRST_RESPONDER`: Creates initial answer with self-reflection and search queries
- `EXECUTE_TOOLS`: Runs web searches using Tavily Search API
- `REVISE`: Incorporates new information and adds proper citations

## Key Features
- **Structured Output**: Uses Pydantic schemas for consistent response formatting
- **Self-Reflection**: Agent critiques its own work identifying missing/superfluous content
- **Web Research**: Automatically generates and executes targeted search queries
- **Citation Management**: Adds numbered references with URLs in revision phase
- **Iteration Control**: Maximum 2 iterations to balance quality vs efficiency
- **Tool Calling**: Leverages OpenAI function calling for structured responses

## Files Structure
```
langgraph_reflexion_agent/
├── chains.py           # LLM chains and prompts for research and revision
├── main.py            # Main graph implementation and execution logic
├── schemas.py         # Pydantic models for structured outputs
├── tool_executor.py   # Web search tool integration with Tavily
├── reflexion_graph.png # Visual representation of the graph
└── __pycache__/       # Python cache files
```

## Technical Implementation

### Schemas (`schemas.py`)
- **`Reflection`**: Captures critique (missing/superfluous content)
- **`AnswerQuestion`**: Initial response with answer, reflection, and search queries
- **`ReviseAnswer`**: Enhanced version with citations and references

### Chains (`chains.py`)
- **First Responder**: Generates initial detailed answer with self-critique
- **Revise Chain**: Incorporates research findings and adds citations
- **Dynamic Prompts**: Uses current timestamp and specialized instructions

### Tool Execution (`tool_executor.py`)
- **Tavily Integration**: Concurrent web search execution
- **Batch Processing**: Runs multiple search queries simultaneously
- **Structured Tools**: Maps to specific schema outputs

## How It Works
1. **Initial Response**: Agent provides ~250-word answer and identifies improvement areas
2. **Search Generation**: Creates 1-3 targeted search queries based on critique
3. **Web Research**: Executes searches concurrently using Tavily API
4. **Revision**: Incorporates findings, adds citations, maintains word limit
5. **Termination**: Stops after 2 tool execution cycles

## Example Use Case
**Input**: "Write about AI-powered SOC / autonomous SOC problem domain, and list startups that do that and raised capital"

**Process**:
1. Initial comprehensive answer about AI-powered Security Operations Centers
2. Self-critique identifying missing startup examples or funding details
3. Web searches for recent SOC startups and funding information
4. Revised answer with specific companies, funding amounts, and numbered citations

## Advanced Features
- **Concurrent Search**: Batch processing of multiple queries for efficiency
- **Citation Formatting**: Automatic numbered reference system
- **Word Limit Enforcement**: Maintains ~250 word constraint while adding value
- **Time-Aware Prompts**: Includes current timestamp for context
