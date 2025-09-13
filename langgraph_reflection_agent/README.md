# LangGraph Reflection Agent

## Overview
This folder contains an implementation of a **reflection agent** using LangGraph that iteratively improves Twitter posts through a critique-and-refine cycle.

## What It Does
The reflection agent operates using two specialized AI agents:
1. **Generation Agent**: Creates/improves Twitter posts based on user input
2. **Reflection Agent**: Acts as a viral Twitter influencer critic, providing detailed feedback and recommendations

## Architecture
The system uses a **MessageGraph** with two main nodes:
- `GENERATE`: Generates or refines Twitter content
- `REFLECT`: Provides critique and improvement suggestions

## Key Features
- **Iterative Improvement**: The agent cycles between generation and reflection up to 6 messages
- **Automatic Termination**: Stops after 6 messages to prevent infinite loops  
- **Specialized Prompts**: 
  - Generation agent acts as a "twitter techie influencer assistant"
  - Reflection agent acts as a "viral twitter influencer" providing detailed critiques
- **Human-like Feedback Loop**: Reflection output is converted to HumanMessage to simulate user feedback

## Files Structure
```
langgraph_reflection_agent/
├── chains.py          # Defines the LLM chains and prompts for both agents
├── main.py           # Main graph implementation and execution logic
├── reflection_graph.png  # Visual representation of the graph structure
└── __pycache__/      # Python cache files
```

## How It Works
1. User provides initial tweet or improvement request
2. **Generate Node**: Creates/improves the tweet
3. **Conditional Logic**: Checks if conversation has exceeded 6 messages
4. **Reflect Node**: Provides detailed critique and recommendations
5. **Loop**: Returns to Generate node with the critique as "user" feedback
6. **Termination**: Ends after 6 total messages

## Technical Details
- Uses OpenAI GPT-4.1 model
- Built with LangChain and LangGraph
- MessageGraph handles message flow between nodes
- Environment variables loaded via python-dotenv

## Example Use Case
Input: A poorly written tweet about LangChain's tool calling feature
Output: An iteratively improved, more engaging and viral-worthy tweet through multiple rounds of generation and critique.
