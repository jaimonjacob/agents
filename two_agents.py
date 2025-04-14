from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
import asyncio
import os
from autogen_agentchat.ui import Console
from dotenv import load_dotenv

load_dotenv()


model_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# Create the primary agent.
Story_writer = AssistantAgent(
    "Story_writer",
    model_client=model_client,
    system_message="You are a helpful AI assistant which write the story for kids. Keep the story short",
)

# Create the critic agent.
Story_reviewer = AssistantAgent(
    "Story_reviewer",
    model_client=model_client,
    system_message="You are a helpful AI assistant which Provides constructive feedback on Kids stories to add a postive impactful ending. Respond with 'APPROVE' to when your feedbacks are addressed.",
)

# Define a termination condition that stops the task if the critic approves.
text_termination = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat([Story_writer, Story_reviewer], termination_condition=text_termination)

  # Define the main asynchronous function
async def main():
    await Console(
        team.run_stream(task="write a story on lion")
    )  # Stream the messages to the console.

# Run the asynchronous function
if __name__ == "__main__":
    asyncio.run(main())