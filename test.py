from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination


load_dotenv()

# Initialize the model client
model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    api_type="gemini",
)

# Update the fetch tool to connect to the MCP server defined in server.py
fetch_mcp_server = StdioServerParams(command="python", args=["server.py"])

async def main():
    # Create an MCP workbench which provides a session to the MCP server.
    async with McpWorkbench(fetch_mcp_server) as workbench:  # type: ignore
        # Create the calculator agent (uses MCP tools)
        calculator_agent = AssistantAgent(
            name="calculator",
            model_client=model_client,
            workbench=workbench,
            reflect_on_tool_use=True,
            system_message="You are a calculator agent. Perform mathematical operations.",
        )

        # Create the greeter agent (does not use MCP tools)
        greeter_agent = AssistantAgent(
            name="greeter",
            model_client=model_client,
            system_message="You are a manager and greeter agent. You greet the user, assign work to other agents, and respond back to user when all is saisifed. Please respond with TERMINATE once all done",
        )

        # Create a team with the calculator and greeter agents
        team = RoundRobinGroupChat([calculator_agent, greeter_agent])
        
        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=4)
        
        team = RoundRobinGroupChat([calculator_agent, greeter_agent], termination_condition=termination)


        # Define a collaborative task for the team
        task = """
        Hellow my name is Jaimon. Can you please help me add 100 and 20.

        """

        # Run the task with the team        
        await Console(team.run_stream(task=task))

# Run the async main function
asyncio.run(main())
