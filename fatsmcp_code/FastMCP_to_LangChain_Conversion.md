# Converting FastMCP Tools to LangChain Tools

## Overview

This document explains the concepts and implementation details behind converting FastMCP (Model Context Protocol) tools into LangChain-compatible tools, enabling their use with LangChain agents and LLMs.

## Table of Contents

1. [Understanding the Components](#understanding-the-components)
2. [Why Conversion is Needed](#why-conversion-is-needed)
3. [Key Concepts](#key-concepts)
4. [Conversion Process](#conversion-process)
5. [Implementation Details](#implementation-details)
6. [Challenges and Solutions](#challenges-and-solutions)

---

## Understanding the Components

### FastMCP Tools

**FastMCP** is a framework for building MCP (Model Context Protocol) servers that expose tools, resources, and prompts. FastMCP tools have the following characteristics:

- **Tool Definition**: Tools are defined using decorators (`@mcp.tool()`) on Python functions
- **Schema Format**: Tools use JSON Schema format for input validation
- **Async Execution**: Tools are called asynchronously via the FastMCP client
- **Result Format**: Results are returned as `CallToolResult` objects containing `TextContent` items

**Example FastMCP Tool:**
```python
@mcp.tool()
def web_search(query: str) -> dict:
    """
    Perform a web search using Tavily API.
    Returns a dictionary of search results.
    """
    # Tool implementation
    return {"results": [...]}
```

### LangChain Tools

**LangChain Tools** are standardized tool interfaces that work with LangChain agents and LLMs. They have:

- **BaseTool Class**: All LangChain tools inherit from `BaseTool`
- **Pydantic Models**: Input validation using Pydantic models
- **Async Support**: Must implement `_arun()` for async execution
- **Schema Definition**: Uses Pydantic models for type checking and validation

**Key Requirements:**
- Must inherit from `BaseTool`
- Must implement `_run()` (sync) or `_arun()` (async)
- Must define `name` and `description`
- Can optionally define `args_schema` (Pydantic model)

---

## Why Conversion is Needed

### Integration Goals

1. **Agent Compatibility**: LangChain agents expect tools in a specific format
2. **LLM Function Calling**: Modern LLMs (like Groq, OpenAI) need tools in a standardized format
3. **Type Safety**: Pydantic models provide runtime type validation
4. **Unified Interface**: Allows using FastMCP tools alongside native LangChain tools

### Use Cases

- **Multi-Tool Agents**: Combine FastMCP tools with other LangChain tools
- **LLM Function Calling**: Enable LLMs to automatically call FastMCP tools
- **Agent Orchestration**: Use FastMCP tools in complex agent workflows

---

## Key Concepts

### 1. Schema Translation

**JSON Schema → Pydantic Model**

FastMCP tools use JSON Schema for input validation:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query"
    }
  },
  "required": ["query"]
}
```

This must be converted to a Pydantic model:
```python
class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
```

### 2. Type Mapping

**JSON Schema Types → Python Types**

| JSON Schema Type | Python Type |
|-----------------|-------------|
| `string` | `str` |
| `integer` | `int` |
| `number` | `float` |
| `boolean` | `bool` |
| `array` | `list` |
| `object` | `dict` or Pydantic model |

### 3. Async Execution Bridge

**FastMCP Client → LangChain Tool**

The conversion creates a bridge that:
- Receives calls from LangChain agents
- Forwards them to the FastMCP client
- Converts results back to LangChain format

### 4. Result Format Conversion

**CallToolResult → String**

FastMCP returns `CallToolResult` objects:
```python
result = await mcp_client.call_tool("web_search", {"query": "AI"})
# result.content[0].text contains JSON string
```

LangChain tools return strings:
```python
async def _arun(self, **kwargs) -> str:
    result = await self._mcp_client.call_tool(...)
    return result.content[0].text  # Convert to string
```

---

## Conversion Process

### Step 1: Extract Tool Metadata

From FastMCP tool definition:
```python
tool.name          # "web_search"
tool.description   # "Perform a web search..."
tool.inputSchema   # JSON Schema dict
```

### Step 2: Parse JSON Schema

Extract properties and required fields:
```python
properties = tool_schema.get("properties", {})
required = tool_schema.get("required", [])
```

### Step 3: Create Pydantic Model

Dynamically create a Pydantic model:
```python
# Map JSON Schema types to Python types
for prop_name, prop_info in properties.items():
    prop_type = map_json_type_to_python(prop_info.get("type"))
    annotations[prop_name] = prop_type
    field_definitions[prop_name] = Field(...)

# Create model class
InputModel = type(
    f"{tool_name}Input",
    (BaseModel,),
    {"__annotations__": annotations, **field_definitions}
)
```

### Step 4: Create LangChain Tool Wrapper

Create a `BaseTool` subclass:
```python
class FastMCPTool(BaseTool):
    def __init__(self, mcp_client, tool_name, tool_description, tool_schema):
        # Store FastMCP client reference
        # Create Pydantic model from schema
        # Initialize BaseTool with name, description, args_schema
```

### Step 5: Implement Async Execution

```python
async def _arun(self, **kwargs) -> str:
    # Call FastMCP tool
    result = await self._mcp_client.call_tool(self._tool_name, kwargs)
    # Extract and format result
    return format_result(result)
```

---

## Implementation Details

### Dynamic Pydantic Model Creation

**Challenge**: We need to create Pydantic models at runtime based on JSON Schema.

**Solution**: Use Python's `type()` function to dynamically create classes:

```python
InputModel = type(
    f"{tool_name}Input",           # Class name
    (BaseModel,),                  # Base class
    {
        "__annotations__": annotations,  # Type hints
        **field_definitions              # Field definitions
    }
)
```

### Storing Client Reference

**Challenge**: Pydantic models don't allow arbitrary attributes.

**Solution**: Use `object.__setattr__()` to bypass Pydantic validation:

```python
object.__setattr__(self, '_mcp_client', mcp_client)
object.__setattr__(self, '_tool_name', tool_name)
```

### Handling Optional Fields

**Required vs Optional**:
```python
if prop_name in required:
    field_definitions[prop_name] = Field(description=...)
else:
    field_definitions[prop_name] = Field(default=None, description=...)
```

### Result Formatting

**JSON to String Conversion**:
```python
result_text = result.content[0].text
try:
    result_data = json.loads(result_text)
    return json.dumps(result_data, indent=2)  # Pretty print
except json.JSONDecodeError:
    return result_text  # Return as-is if not JSON
```

---

## Challenges and Solutions

### Challenge 1: Pydantic v2 Compatibility

**Problem**: Pydantic v2 has stricter validation and doesn't allow arbitrary attributes.

**Solution**: 
- Use `object.__setattr__()` for private attributes
- Properly define all fields in the model
- Use `model_config` if needed for extra fields

### Challenge 2: Empty or Invalid Schemas

**Problem**: Some tools might have empty or malformed schemas.

**Solution**:
```python
if not tool_schema:
    tool_schema = {"type": "object", "properties": {}, "required": []}
elif "type" not in tool_schema:
    tool_schema = {**tool_schema, "type": "object"}
```

### Challenge 3: Type Mapping Complexity

**Problem**: JSON Schema has more types than basic Python types.

**Solution**: Create a mapping function:
```python
def map_json_type_to_python(json_type: str):
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        # Handle arrays and objects as needed
    }
    return type_map.get(json_type, str)  # Default to string
```

### Challenge 4: Async Context Management

**Problem**: FastMCP client needs to stay connected during tool execution.

**Solution**: 
- Keep client in async context manager
- Store client reference in tool wrapper
- Ensure client connection is maintained

### Challenge 5: Error Handling

**Problem**: Tool calls can fail for various reasons.

**Solution**:
```python
async def _arun(self, **kwargs) -> str:
    try:
        result = await self._mcp_client.call_tool(...)
        return format_result(result)
    except Exception as e:
        return f"Error calling tool: {str(e)}"
```

---

## Complete Example

### FastMCP Tool Definition
```python
@mcp.tool()
def web_search(query: str) -> dict:
    """Perform a web search"""
    # Implementation
    return {"results": [...]}
```

### Converted LangChain Tool
```python
class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")

class FastMCPTool(BaseTool):
    name = "web_search"
    description = "Perform a web search"
    args_schema = WebSearchInput
    
    async def _arun(self, query: str) -> str:
        result = await self._mcp_client.call_tool("web_search", {"query": query})
        return result.content[0].text
```

### Usage with LangChain Agent
```python
# Convert tools
langchain_tools = convert_fastmcp_tools_to_langchain(client, tools_list)

# Create agent
model = ChatGroq(...)
model_with_tools = model.bind_tools(langchain_tools)
agent = create_agent(model_with_tools, langchain_tools)

# Use agent
result = await agent.ainvoke({"messages": [HumanMessage(content="Search for AI")]})
```

---

## Benefits of This Approach

1. **Reusability**: FastMCP tools can be used in any LangChain workflow
2. **Type Safety**: Pydantic models provide runtime validation
3. **LLM Integration**: Tools work seamlessly with function calling
4. **Flexibility**: Can combine FastMCP tools with other LangChain tools
5. **Maintainability**: Single source of truth (FastMCP server) for tool definitions

---

## Best Practices

1. **Schema Validation**: Always validate and normalize schemas before conversion
2. **Error Handling**: Provide meaningful error messages when tool calls fail
3. **Type Safety**: Use proper type hints and Pydantic models
4. **Documentation**: Ensure tool descriptions are clear and helpful for LLMs
5. **Testing**: Test tools individually before using in agents
6. **Connection Management**: Properly manage FastMCP client connections

---

## Conclusion

Converting FastMCP tools to LangChain tools enables seamless integration between FastMCP servers and LangChain agents. The key is understanding:

- **Schema Translation**: JSON Schema → Pydantic Models
- **Type Mapping**: JSON types → Python types
- **Execution Bridge**: LangChain calls → FastMCP execution
- **Result Formatting**: CallToolResult → String

This conversion allows developers to leverage FastMCP's simplicity while benefiting from LangChain's powerful agent capabilities.


