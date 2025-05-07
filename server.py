# server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP(
    # Define the server address and port
    address="localhost",
    port=5000,
    # Define the server name and description
    name="DemoServer",
    description="A simple MCP server for demonstration purposes",
)


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

if __name__ == "__main__":
    try:# Start the MCP server
        mcp.run()
        print("MCP server is running...")
    except Exception as e:
        print(f"Error starting MCP server: {e}")