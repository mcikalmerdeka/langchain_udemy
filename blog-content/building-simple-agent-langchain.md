# **Building Your First AI Agent with LangChain: A Practical Guide**

**Author:** Muhammad Cikal Merdeka
**Date:** January 2025
**Reading Time:** 15 minutes

## Introduction

You've heard about AI agents. Maybe you've seen demos of ChatGPT plugins or autonomous systems that can browse the web, write code, and solve complex problems. You wondered how they actually *work*.

Maybe you tried to read the documentation. Got hit with a wall of concepts. Chains, agents, tools, memory. It felt like opaque, magical black boxes.

**Here's the truth: They're not black boxes. They are elegant orchestrations of LLMs, tools, and decision-making logic that you can understand and build yourself.**

An AI agent is fundamentally just a program that can:

1. **Think** - Use an LLM to reason about what to do
2. **Act** - Execute tools to interact with the world
3. **Observe** - Process the results and decide what to do next
4. **Repeat** - Continue until the task is complete

**In this guide, we'll build a functional AI agent from scratch using LangChain.**

You'll see *why* agents work by understanding each component. Once you've built the basic structure, scaling to more complex agents becomes obvious just by adding more tools and refining the prompts.

### What is LangChain?

**LangChain** is a framework designed to simplify building applications powered by large language models (LLMs). Think of it as a toolkit that handles the repetitive, complex parts of working with LLMs so you can focus on building your application logic.

**Why LangChain matters for agents:**

- **Abstracts the complexity** - Instead of manually formatting prompts and parsing outputs, LangChain provides clean interfaces
- **Provides pre-built components** - Tools, memory systems, and agent executors are ready to use
- **Handles the orchestration** - The framework manages the agent loop, tool calling, and error handling
- **Extensible** - You can easily add custom tools and integrate with various LLM providers
- **Built on LangGraph** - Modern LangChain agents (as of 2025) are built on top of LangGraph, giving you production-ready state management under the hood

Without LangChain, you'd need to write hundreds of lines of boilerplate code just to get a basic agent running. With it, you can build a functional agent in under 50 lines.

---

## Part 1: Understanding the Agent Architecture

Before we write any code, let's understand what an agent actually is.

### 1.1. The Core Concept

An **agent** is a system that uses an LLM as a **reasoning engine** to decide which actions to take. Unlike a simple chatbot that just responds to queries, an agent can:

- Break down complex tasks into steps
- Choose and use external tools
- Maintain context across multiple interactions (really important!)
- Self-correct when things go wrong

The Analogy: Think of an agent as a chef preparing a recipe. You ask for "chocolate chip cookies." The chef:

1. Realizes they need to check the pantry for ingredients
2. Finds flour, sugar, and butter but no chocolate chips
3. Observes they're missing a key ingredient
4. Decides to use the shopping tool to order chocolate chips
5. Once delivered, combines all ingredients and bakes
6. Serves you the finished cookies

### 1.2. The Agent Loop (ReAct Pattern)

The most common agent pattern is called **ReAct** (Reasoning + Acting). It follows this loop:

```
1. THOUGHT: "What should I do next?"
2. ACTION: Execute a tool
3. OBSERVATION: See the result
4. THOUGHT: "What does this mean? Am I done?"
5. Repeat or FINAL ANSWER
```

This is not an analogy, this is the literal structure of how agents work. We will see it in this next part.

---

## Part 2: Setting Up Your Environment

### 2.1. Installation

First, initialize a new project with uv and install the required packages:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new project
uv init langchain-agent

# Navigate to project directory
cd langchain-agent

# Create and activate virtual environment
uv venv

# Install dependencies
uv add langchain langchain-openai python-dotenv

# Optional: Install langgraph for advanced memory features (Part 8)
uv add langgraph
```

**Why uv?** It's significantly faster than pip/pipenv/poetry and handles virtual environments automatically.

### 2.2. API Keys

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2.3. Basic Imports

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# Load environment variables
load_dotenv()
```

---

## Part 3: Building Your First Tool

Tools are the "hands" of your agent—they allow it to interact with the world. Let's create tools that access external systems the LLM can't reach natively.

### 3.1. Creating a Warehouse Inventory Tool

```python
from langchain_core.tools import tool

@tool
def check_inventory(product_id: str) -> str:
    """Checks product inventory levels and locations. Input should be a product ID like 'PROD-123'.
  
    Args:
        product_id: The product ID to check (e.g., "PROD-123")
  
    Returns:
        Inventory status and location information
    """
    try:
        # In production, this would be your actual warehouse API
        # For demo, we'll simulate the API response
        inventory_data = {
            "PROD-123": {"quantity": 45, "location": "Aisle 3, Shelf B2", "status": "In Stock"},
            "PROD-456": {"quantity": 0, "location": "N/A", "status": "Out of Stock"},
            "PROD-789": {"quantity": 12, "location": "Aisle 1, Shelf C1", "status": "In Stock"}
        }

        if product_id not in inventory_data:
            return f"Product {product_id} not found in inventory system."

        item = inventory_data[product_id]
        return f"Product {product_id}: {item['quantity']} units available at {item['location']} ({item['status']})"

    except Exception as e:
        return f"Error accessing inventory: {str(e)}"
```

**Key Points:**

- The `@tool` decorator automatically converts your function into a LangChain tool
- The docstring's first line becomes the tool description—this is crucial for the LLM to understand when to use the tool
- Type hints help LangChain understand the expected inputs

### 3.2. Creating a Customer Order Tool

```python
@tool
def check_order_status(order_id: str) -> str:
    """Checks order status and shipping information. Input should be an order ID like 'ORD-2024-001'.
  
    Args:
        order_id: The order number to check (e.g., "ORD-2024-001")
  
    Returns:
        Order status and shipping information
    """
    try:
        # Simulate order management system API
        orders_db = {
            "ORD-2024-001": {
                "status": "Shipped",
                "tracking": "1Z999AA1234567890",
                "estimated_delivery": "2024-01-20",
                "items": ["PROD-123", "PROD-456"]
            },
            "ORD-2024-002": {
                "status": "Processing",
                "tracking": None,
                "estimated_delivery": "2024-01-22",
                "items": ["PROD-789"]
            }
        }

        if order_id not in orders_db:
            return f"Order {order_id} not found in system."

        order = orders_db[order_id]
        status = f"Order {order_id} status: {order['status']}"

        if order['tracking']:
            status += f"\nTracking number: {order['tracking']}"
            status += f"\nEstimated delivery: {order['estimated_delivery']}"

        return status

    except Exception as e:
        return f"Error accessing order system: {str(e)}"
```

---

## Part 4: Creating the Agent

Now we assemble our tools into an agent.

### 4.1. Initialize the LLM

```python
# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,  # Lower temperature for more consistent reasoning
    api_key=os.getenv("OPENAI_API_KEY")
)
```

**Why temperature=0?** For agents, we want consistent, logical reasoning rather than creative responses.

### 4.2. Combine Tools

```python
# Create a list of all available tools
tools = [check_inventory, check_order_status]
```

### 4.3. Get the Agent Prompt

LangChain provides pre-built prompts optimized for agents. You can check the prompt here: https://smith.langchain.com/hub/hwchase17/react

```python
# Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")
```

This prompt template includes:

- Instructions for the ReAct pattern
- How to format tool calls
- When to provide a final answer

### 4.4. Create the Agent

Now we create the agent using the modern `create_react_agent` function (which is now built on top of LangGraph under the hood):

```python
# Create the agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Wrap it in an executor
# AgentExecutor manages the agent's execution loop, handling tool calls, parsing responses, and coordinating between the LLM and tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Shows the agent's thinking process
    handle_parsing_errors=True,  # Gracefully handle errors
    max_iterations=5  # Prevent infinite loops
)
```

**Key Parameters:**

- `verbose=True`: Lets you see the agent's thought process
- `handle_parsing_errors=True`: Makes the agent more robust
- `max_iterations=5`: Safety limit to prevent runaway execution

**Note**: As of 2025, `langchain.agents.create_react_agent` is built on top of LangGraph, giving you the benefits of modern state management while maintaining a simple API

---

## Part 5: Running Your Agent

### 5.1. Simple Query

```python
# Test with a warehouse inventory check
response = agent_executor.invoke({
    "input": "Check inventory for product PROD-123"
})

print(response["output"])
```

**Output:**

```
> Entering new AgentExecutor chain...
I need to check the warehouse inventory for PROD-123.

Action: check_inventory
Action Input: PROD-123

Observation: Product PROD-123: 45 units available at Aisle 3, Shelf B2 (In Stock)
Thought: I now know the inventory status
Final Answer: Product PROD-123 has 45 units available at Aisle 3, Shelf B2 and is currently in stock.

> Finished chain.
Product PROD-123 has 45 units available at Aisle 3, Shelf B2 and is currently in stock.
```

### 5.2. Multi-Step Query

```python
# Test with a query requiring multiple tools
response = agent_executor.invoke({
    "input": "Check order ORD-2024-001 status and inventory for PROD-456"
})

print(response["output"])
```

**Output:**

```
> Entering new AgentExecutor chain...
I need to check both the order status and inventory for different products.

Action: check_order_status
Action Input: ORD-2024-001

Observation: Order ORD-2024-001 status: Shipped
Tracking number: 1Z999AA1234567890
Estimated delivery: 2024-01-20
Thought: Now I need to check inventory for PROD-456

Action: check_inventory
Action Input: PROD-456

Observation: Product PROD-456: 0 units available at N/A (Out of Stock)
Thought: I have both pieces of information
Final Answer: Order ORD-2024-001 has been shipped with tracking number 1Z999AA1234567890 and estimated delivery on 2024-01-20. Product PROD-456 is currently out of stock.

> Finished chain.
```

---

## Part 6: Understanding What Just Happened

Let's break down the agent's execution based on the output above:

### 6.1. The Thought Process

```
THOUGHT → ACTION → OBSERVATION → THOUGHT → ACTION → OBSERVATION → FINAL ANSWER
```

This is the ReAct loop in action. The agent:

1. **Thought**: Analyzed the query and decided to check order status first
2. **Action**: Chose the OrderStatus tool
3. **Observation**: Received the order information including tracking details
4. **Thought**: Realized it needed to check inventory for another product
5. **Action**: Chose the WarehouseInventory tool
6. **Observation**: Received the inventory status showing out of stock
7. **Final Answer**: Combined both results into a comprehensive response

### 6.2. The Prompt Behind the Scenes

The agent uses a prompt template that looks like this (simplified):

```
Answer the following questions as best you can. You have access to the following tools:

check_inventory: Checks product inventory levels and locations...
check_order_status: Checks order status and shipping information...

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [check_inventory, check_order_status]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: {agent_scratchpad}
```

---

## Part 7: Building a More Practical Agent

Let's create a more useful agent with real-world business tools.

### 7.1. Adding a Customer Lookup Tool

```python
@tool
def lookup_customer(customer_id: str) -> str:
    """Looks up customer information by ID. Input should be a customer ID like 'CUST-001'.
  
    Args:
        customer_id: The customer ID to look up (e.g., "CUST-001")
  
    Returns:
        Customer information from the CRM system
    """
    try:
        # Simulate CRM system API
        customers_db = {
            "CUST-001": {
                "name": "John Smith",
                "email": "john.smith@email.com",
                "phone": "+1-555-0123",
                "status": "Active",
                "join_date": "2023-01-15"
            },
            "CUST-002": {
                "name": "Sarah Johnson",
                "email": "sarah.j@email.com",
                "phone": "+1-555-0456",
                "status": "Premium",
                "join_date": "2022-11-20"
            }
        }

        if customer_id not in customers_db:
            return f"Customer {customer_id} not found in CRM system."

        customer = customers_db[customer_id]
        return f"Customer {customer_id}: {customer['name']}\nEmail: {customer['email']}\nPhone: {customer['phone']}\nStatus: {customer['status']}\nMember since: {customer['join_date']}"

    except Exception as e:
        return f"Error accessing customer data: {str(e)}"
```

### 7.2. Adding a Product Pricing Tool

```python
@tool
def get_product_price(product_id: str) -> str:
    """Gets current product pricing and discounts. Input should be a product ID like 'PROD-123'.
  
    Args:
        product_id: The product ID to check pricing for (e.g., "PROD-123")
  
    Returns:
        Current pricing information including any discounts
    """
    try:
        # Simulate product pricing API
        pricing_db = {
            "PROD-123": {"price": 29.99, "currency": "USD", "discount": 0},
            "PROD-456": {"price": 149.99, "currency": "USD", "discount": 10},
            "PROD-789": {"price": 79.99, "currency": "USD", "discount": 0}
        }

        if product_id not in pricing_db:
            return f"Product {product_id} not found in pricing system."

        product = pricing_db[product_id]
        price_str = f"Product {product_id}: ${product['price']} {product['currency']}"

        if product['discount'] > 0:
            price_str += f" ({product['discount']}% discount applied)"

        return price_str

    except Exception as e:
        return f"Error accessing pricing data: {str(e)}"
```

### 7.3. Updated Agent

```python
# Combine all tools for comprehensive business operations
tools = [check_inventory, check_order_status, lookup_customer, get_product_price]

# Create new agent with expanded business toolset
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)
```

---

## Part 8: Advanced Patterns

### 8.1. Adding Memory

To make your agent remember previous interactions, we can use **LangGraph's modern state management** while keeping the `langchain.agents` implementation. Since `ConversationBufferMemory` is deprecated, here's the recommended approach:

**Option 1: Using LangGraph State Management (Recommended)**

Build a simple LangGraph wrapper around your agent to get proper memory with `MemorySaver`:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# Create memory checkpoint
memory = MemorySaver()

# Initialize the graph
builder = StateGraph(state_schema=MessagesState)

# Define the agent node - wraps your AgentExecutor
def agent_node(state: MessagesState):
    """Node that calls your agent with conversation history."""
    # Get the latest user message
    user_message = state["messages"][-1].content
    
    # Invoke your regular AgentExecutor
    response = agent_executor.invoke({"input": user_message})
    
    # Return the response as a message
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response["output"])]}

# Build the graph
builder.add_node("agent", agent_node)
builder.add_edge(START, "agent")

# Compile with memory
agent_with_memory = builder.compile(checkpointer=memory)

# Use with thread_id for conversation sessions
config = {"configurable": {"thread_id": "user-123"}}

# First interaction
response1 = agent_with_memory.invoke(
    {"messages": [HumanMessage(content="Check inventory for PROD-123")]},
    config=config
)
print("Bot:", response1["messages"][-1].content)

# Second interaction - agent remembers context
response2 = agent_with_memory.invoke(
    {"messages": [HumanMessage(content="What about PROD-456?")]},
    config=config
)
print("Bot:", response2["messages"][-1].content)
```

**Option 2: Manual State Management (Simple Alternative)**

For simpler use cases, manually track conversation history:

```python
# Store conversation history manually
conversation_history = []

def invoke_agent_with_memory(user_input: str):
    """Invoke agent with conversation history context."""
    # Add user message to history
    conversation_history.append(f"User: {user_input}")
    
    # Build context from recent history (last 5 exchanges)
    recent_history = "\n".join(conversation_history[-10:])
    
    # Create input with context
    full_input = f"""Previous conversation:
{recent_history}

Current request: {user_input}"""
    
    # Invoke agent
    response = agent_executor.invoke({"input": full_input})
    output = response["output"]
    
    # Add response to history
    conversation_history.append(f"Assistant: {output}")
    
    return output

# Usage
print(invoke_agent_with_memory("Check inventory for PROD-123"))
print(invoke_agent_with_memory("What about PROD-456?"))  # Agent sees previous exchange
```

**Key Points:**

- **Option 1**: Best for production - proper state management with `MemorySaver`
- **Option 2**: Simpler but less scalable - good for prototypes
- **`thread_id`**: Identifies separate conversation sessions (different users)
- **LangGraph wrapper**: Lets you use modern memory while keeping `AgentExecutor`
- Your core agent logic remains unchanged with `langchain.agents`

### 8.2. Custom Tool with Pydantic

For tools with multiple parameters, you can use Pydantic's `Field` for better validation and documentation. Here's an example of a weather forecast tool:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import Literal, Optional

class WeatherInput(BaseModel):
    """Input schema for weather forecast tool."""
    city: str = Field(
        description="City name (e.g., 'New York', 'London')"
    )
    country_code: str = Field(
        default="US",
        description="Two-letter country code (e.g., 'US', 'UK', 'FR')",
        max_length=2,
        min_length=2
    )
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    days: int = Field(
        default=3,
        description="Number of forecast days (1-7)",
        ge=1,  # greater than or equal to 1
        le=7   # less than or equal to 7
    )
    include_humidity: bool = Field(
        default=False,
        description="Include humidity data in the forecast"
    )
    alert_threshold: Optional[float] = Field(
        default=None,
        description="Temperature threshold for alerts (optional)",
        gt=-50,  # greater than -50
        lt=60    # less than 60
    )

@tool(args_schema=WeatherInput)
def get_weather_forecast(
    city: str,
    country_code: str = "US",
    units: str = "celsius",
    days: int = 3,
    include_humidity: bool = False,
    alert_threshold: Optional[float] = None
) -> str:
    """Gets weather forecast for a specified city with various options.
    
    Args:
        city: City name
        country_code: Two-letter country code
        units: Temperature units (celsius/fahrenheit)
        days: Number of days to forecast (1-7)
        include_humidity: Whether to include humidity data
        alert_threshold: Optional temperature alert threshold
    
    Returns:
        Weather forecast information
    """
    try:
        # Simulate weather API call
        forecast = f"🌤️ Weather Forecast for {city}, {country_code}\n\n"
        forecast += f"Forecast for next {days} day(s) in {units}:\n\n"
        
        for day in range(1, days + 1):
            temp = 20 + (day * 2)  # Simulated temperature
            forecast += f"Day {day}: {temp}°{'C' if units == 'celsius' else 'F'}"
            
            if include_humidity:
                humidity = 60 + (day * 5)
                forecast += f", Humidity: {humidity}%"
            
            if alert_threshold and temp > alert_threshold:
                forecast += " ⚠️ ALERT: Above threshold!"
            
            forecast += "\n"
        
        return forecast
        
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# Add to your agent's tools
tools_with_weather = [check_inventory, check_order_status, get_weather_forecast]
agent = create_react_agent(llm=llm, tools=tools_with_weather, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools_with_weather, verbose=True)

# Example usage
response = agent_executor.invoke({
    "input": "What's the weather forecast for London, UK for the next 5 days in fahrenheit with humidity data?"
})
```

**Key Pydantic Field Features:**

- `description`: Helps the LLM understand what the parameter is for
- `default`: Provides default values for optional parameters
- `min_length` / `max_length`: Validates string or list length
- `ge` / `le` / `gt` / `lt`: Numeric constraints (greater/less than)
- `Literal`: Restricts to specific allowed values
- `Optional`: Makes parameters optional with None as default

### 8.3. Error Handling

```python
def safe_agent_call(query: str) -> str:
    """Safely execute agent with error handling."""
    try:
        response = agent_executor.invoke({"input": query})
        return response["output"]
    except Exception as e:
        return f"Agent error: {str(e)}"

# Usage
result = safe_agent_call("Check inventory for product PROD-999")
print(result)
```

---

## Part 9: Best Practices

### 9.1. Tool Design Principles

1. **Clear Descriptions**: The LLM relies on descriptions to choose tools

   ```python
   # Good
   description="Calculates mathematical expressions. Input: '2+2' or '10*5'"

   # Bad
   description="Does math"
   ```
2. **Single Responsibility**: Each tool should do one thing well

   ```python
   # Good: Separate tools
   - Calculator: Math operations
   - Search: Information retrieval

   # Bad: One tool that does everything
   - SuperTool: Math, search, file operations, etc.
   ```
3. **Error Handling**: Always return informative error messages

   ```python
   try:
       result = perform_operation()
       return f"Success: {result}"
   except Exception as e:
       return f"Error: {str(e)}. Please try again with valid input."
   ```

### 9.2. Prompt Engineering for Agents

The quality of your agent depends heavily on the prompt. Key elements:

1. **Clear Instructions**: Tell the agent exactly what to do
2. **Tool Descriptions**: Detailed descriptions of when to use each tool
3. **Output Format**: Specify how to format the final answer
4. **Examples**: Include few-shot examples for complex tasks

### 9.3. Debugging Tips

```python
# Enable verbose mode to see agent's thinking
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Shows full reasoning chain
    return_intermediate_steps=True  # Returns all steps
)

# Inspect intermediate steps
response = agent_executor.invoke({"input": "Your query"})
print("Final output:", response["output"])
print("\nIntermediate steps:", response.get("intermediate_steps"))

# You can also stream the agent's execution
for step in agent_executor.stream({"input": "Check inventory for PROD-123"}):
    print(step)
```

---

## Part 10: Customizing Agent Behavior

You can customize your agent's behavior by modifying the prompt or adding custom logic:

### Custom Prompts

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Create a custom prompt with specific instructions
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a warehouse management assistant. 
    Always be concise and professional.
    When checking inventory, also suggest reorder if stock is low (below 20 units).
    Format all responses in a clear, structured way."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Use the custom prompt
agent = create_react_agent(llm=llm, tools=tools, prompt=custom_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### Adding Custom Callbacks

```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"🔧 Starting tool: {serialized.get('name')}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"✅ Tool finished: {output[:50]}...")

# Use callbacks for monitoring
response = agent_executor.invoke(
    {"input": "Check inventory for PROD-123"},
    callbacks=[CustomCallback()]
)
```

**When you need more complex behavior:**

For advanced use cases like parallel tool execution, conditional branching, or multi-agent systems, consider using LangGraph directly to build custom agent workflows. But for most applications, `AgentExecutor` with custom prompts and callbacks is sufficient.

---

## Part 11: Real-World Example

Let's build a practical customer service agent that handles real business scenarios:

```python
# Customer service agent for e-commerce business
def create_customer_service_agent():
    """Creates an agent that can handle customer inquiries about orders, products, and inventory."""

    # Combine all business tools
    business_tools = [check_inventory, check_order_status, lookup_customer, get_product_price]

    # Create agent with business-focused tools
    business_agent = create_react_agent(llm=llm, tools=business_tools, prompt=prompt)
    business_executor = AgentExecutor(
        agent=business_agent,
        tools=business_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )

    return business_executor

# Example usage scenarios
customer_service_agent = create_customer_service_agent()

# Scenario 1: Order status inquiry
print("=== Customer Order Status Scenario ===")
response = customer_service_agent.invoke({
    "input": "My customer John Smith (CUST-001) is asking about their order ORD-2024-001. Can you check the status and also tell me if PROD-456 is in stock for a potential replacement?"
})
print(response["output"])

# Scenario 2: Product inquiry with pricing
print("\n=== Product & Pricing Scenario ===")
response = customer_service_agent.invoke({
    "input": "A customer wants to know about PROD-123 - what's the current price and is it available in our warehouse?"
})
print(response["output"])

# Scenario 3: Complex multi-step inquiry
print("\n=== Complex Business Scenario ===")
response = customer_service_agent.invoke({
    "input": "Help me process a potential return: Customer CUST-001 ordered PROD-456 in order ORD-2024-002, but it's showing as out of stock. What's the current status of their order and should we offer PROD-789 as a replacement? Also check the pricing for both products."
})
print(response["output"])
```

---

## Conclusion: From Simple Agent to Production

You've now built a functional AI agent from scratch. You understand:

- **The ReAct Loop**: How agents think, act, and observe
- **Tool Creation**: How to give agents capabilities
- **Agent Execution**: How the pieces fit together
- **Best Practices**: How to make agents reliable

### The Path Forward

The agent you built is the foundation. To scale to production:

1. **Add More Tools**: Integrate APIs, databases, file systems
2. **Improve Prompts**: Fine-tune for your specific use case
3. **Add Memory**: Enable multi-turn conversations
4. **Error Handling**: Make it robust for edge cases
5. **Monitoring**: Track agent performance and costs

### Key Takeaways

- Agents are not magic—they're LLMs + Tools + Logic
- The ReAct pattern is universal across agent frameworks
- Tool descriptions are as important as the tools themselves
- Start simple, then add complexity as needed

**The difference between a simple agent and a production system is not one of kind, but one of scale and refinement.**

You now understand the engine. You're ready to build agents that solve real problems.

---

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/modules/agents/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain Hub](https://smith.langchain.com/hub)
- [Agent Examples Repository](https://github.com/langchain-ai/langchain/tree/master/docs/docs/modules/agents)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/): For building complex agentic AI systems, LangGraph is often preferred over basic LangChain agents. It excels in creating stateful, multi-agent workflows with features like cyclic graphs, persistent state management, and better handling of intricate logic—ideal for production-scale applications requiring coordination and scalability.

---