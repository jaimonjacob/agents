import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

# Initialize the model client
model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    api_type="gemini",
)

# Define the MCP server parameters
fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])

async def main():
    try:
        # Fetch tools from the MCP server
        tools = await mcp_server_tools(fetch_mcp_server)
        
        # Initialize the assistant agent
        agent = AssistantAgent(
            name="fetcher",
            model_client=model_client,
            tools=tools,
            reflect_on_tool_use=True,  # type: ignore
        )
        
        # Execute the task and summarize the content of the URL
        result = await agent.run(task="Summarize the content of https://en.wikipedia.org/wiki/Seattle")
        print(result.messages[-1].content)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
