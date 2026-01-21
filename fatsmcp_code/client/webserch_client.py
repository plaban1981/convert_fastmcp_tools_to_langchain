from fastmcp.client.transports import StdioTransport
from fastmcp import Client
import asyncio
import json
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

transport = StdioTransport(
    command="python",
    args=["-u", r"C:\Users\nayak\Documents\Master_MCP\fatsmcp_code\server\webserch_server.py"]
)
client = Client(transport)

# Custom LangChain tool wrapper for FastMCP tools
class FastMCPTool(BaseTool):
    """Wrapper to convert FastMCP tools to LangChain tools"""
    
    def __init__(self, mcp_client: Client, tool_name: str, tool_description: str, tool_schema: dict):
        # Store client and tool info using object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, '_mcp_client', mcp_client)
        object.__setattr__(self, '_tool_name', tool_name)
        object.__setattr__(self, '_tool_schema', tool_schema)
        
        # Extract parameters from schema
        properties = tool_schema.get("properties", {}) if tool_schema else {}
        required = tool_schema.get("required", []) if tool_schema else []
        
        # Create Pydantic model for input validation
        InputModel = None
        
        if properties:
            field_definitions = {}
            annotations = {}
            
            for prop_name, prop_info in properties.items():
                # Determine Python type
                prop_type = str  # Default to string
                if prop_info.get("type") == "integer":
                    prop_type = int
                elif prop_info.get("type") == "number":
                    prop_type = float
                elif prop_info.get("type") == "boolean":
                    prop_type = bool
                
                # Create field with description
                field_description = prop_info.get("description", "")
                if prop_name in required:
                    field_definitions[prop_name] = Field(description=field_description)
                else:
                    field_definitions[prop_name] = Field(default=None, description=field_description)
                
                annotations[prop_name] = prop_type
            
            # Create dynamic Pydantic model class only if we have properties
            if annotations:
                InputModel = type(
                    f"{tool_name}Input",
                    (BaseModel,),
                    {
                        "__annotations__": annotations,
                        **field_definitions
                    }
                )
        
        # Initialize BaseTool
        init_kwargs = {
            "name": tool_name,
            "description": tool_description or f"Tool: {tool_name}"
        }
        if InputModel:
            init_kwargs["args_schema"] = InputModel
        
        super().__init__(**init_kwargs)
    
    def _run(self, **kwargs) -> str:
        """Synchronous execution (not used for async)"""
        raise NotImplementedError("Use _arun for async execution")
    
    async def _arun(self, **kwargs) -> str:
        """Async execution - calls FastMCP tool"""
        try:
            # Call the FastMCP tool
            result = await self._mcp_client.call_tool(self._tool_name, kwargs)
            
            # Extract text from result
            if result.content:
                result_text = result.content[0].text
                # Try to parse as JSON, if it fails return as string
                try:
                    result_data = json.loads(result_text)
                    return json.dumps(result_data, indent=2)
                except json.JSONDecodeError:
                    return result_text
            return "No result returned"
        except Exception as e:
            return f"Error calling tool: {str(e)}"

def convert_fastmcp_tools_to_langchain(mcp_client: Client, tools_list: list) -> list:
    """Convert FastMCP tools to LangChain tools"""
    langchain_tools = []
    
    for tool in tools_list:
        langchain_tool = FastMCPTool(
            mcp_client=mcp_client,
            tool_name=tool.name,
            tool_description=tool.description or f"Tool: {tool.name}",
            tool_schema=tool.inputSchema or {}
        )
        langchain_tools.append(langchain_tool)
    
    return langchain_tools

async def efficient_multiple_operations(query: str):
    async with client:
        await client.ping()
        print("Ping successful!")
    
    async with client:
        # Get tools from FastMCP
        tools_list = await client.list_tools()
        print(f"Found {len(tools_list)} tools")
        
        # Debug: Print tool schema
        for tool in tools_list:
            print(f"Tool: {tool.name}")
            print(f"Description: {tool.description}")
            print(f"Schema: {tool.inputSchema}")
        
        # Convert to LangChain tools
        langchain_tools = convert_fastmcp_tools_to_langchain(client, tools_list)
        print(f"Converted {len(langchain_tools)} tools to LangChain format")
        
        # Debug: Print LangChain tool info
        for tool in langchain_tools:
            print(f"LangChain Tool: {tool.name}")
            print(f"Tool args_schema: {tool.args_schema}")
            if tool.args_schema:
                print(f"Schema fields: {tool.args_schema.model_fields}")
        
        # Create model and agent
        model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=1000)
        
        # Bind tools to model
        model_with_tools = model.bind_tools(langchain_tools)
        
        # Create agent
        agent = create_agent(model_with_tools, langchain_tools)
        
        # Run agent
        from langchain_core.messages import HumanMessage
        try:
            agent_input = {"messages": [HumanMessage(content=query)]}
            agent_result = await agent.ainvoke(agent_input)
            print("Agent result:", agent_result["messages"][-1].content)
        except Exception as e:
            print(f"Error with agent (this might be a Groq API issue): {e}")
            print("Trying direct tool call instead...")
            # Fallback: direct tool call
            result = await client.call_tool("web_search", {"query": query})
            if result.content:
                result_text = result.content[0].text
                result_data = json.loads(result_text)
                print("Direct tool result:", json.dumps(result_data, indent=2)[:500])  # First 500 chars
    
    # Direct tool call (alternative approach)
    async with client:
        print("\n" + "="*50)
        print("Direct tool call (alternative approach):")
        print("="*50)
        result = await client.call_tool("web_search", {"query": query})
        
        # Access the text content - it's a JSON string that needs to be parsed
        if result.content:
            result_text = result.content[0].text
            result_data = json.loads(result_text)
            print("Direct call result keys:", result_data.keys())
            if "results" in result_data:
                print(f"Number of results: {len(result_data['results'])}")
                if result_data["results"]:
                    print("First result:", result_data["results"][0])

if __name__ == "__main__":
    query = input("Enter a query: ")
    asyncio.run(efficient_multiple_operations(query))