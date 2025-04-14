from autogen_ext.models.openai import OpenAIChatCompletionClient
# from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
import asyncio
import os
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
from autogen_core.tools import FunctionTool
import requests
import streamlit as st

# Streamlit UI
st.title("AutoGen Chat Agents")
st.write("This app visualizes the conversation between agents working collaboratively.")

load_dotenv()

model_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash-8b",    
    api_key=os.getenv("GEMINI_API_KEY"),
    api_type="gemini",
)

# model_client = OllamaChatCompletionClient(
#     model="llama3.2:1b",        
# )

def fetch_random_joke():
    """Fetches a random joke from the official-joke-api."""
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=10) # Add timeout
        response.raise_for_status() # Raise an exception for bad status codes
        joke = response.json()
        return f"{joke['setup']} - {joke['punchline']}"
    except requests.exceptions.RequestException as e:
        return f"Failed to fetch a joke: {e}"
    except Exception as e: # Catch other potential errors like JSON decoding
        return f"An error occurred while processing the joke: {e}"

def calculate_text_length(text: str) -> str:
    return f"The text length is {len(text)} characters."

length_tool = FunctionTool(calculate_text_length, description="Calculate length of characters in the Joke")
fetch_tool =  FunctionTool(fetch_random_joke, description="Fetch a random joke from an API")

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    system_message="""
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        Joke_writer: Writes the joke and make corrections.
        Joke_reviewer: Checks if the joke is suitable for audience. It doesn't write the joke, only provide feedback and improvements.
        Joke_length_checker: Checks if the joke length is under 200 characters in length. It doesn't write the joke, only provide to suggestion on the length.        

    You only plan and delegate tasks - you do not execute them yourself. You can engage team members multiple times so that a perfect Joke is provided.

    When assigning tasks, use this format:
    1. <agent>: <task>

    After assigning tasks, wait for responses from the agents and ensure all subtasks are completed. After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)

Joke_writer = AssistantAgent(
    "Joke_writer",
    model_client=model_client,
    system_message="You are a helpful AI assistant which firest fetches a random joke using 'fetch_random_joke' tool. And then edit it appropriately",
    tools=[fetch_tool],
)

# Create the Reviewer agent.
Joke_reviewer = AssistantAgent(
    "Joke_reviewer",
    model_client=model_client,
    system_message="You are a helpful AI assistant which checks if the joke is suitable for all audience and give feedback so that the joke doesnt offend anyone" 
)

Joke_length_checker = AssistantAgent(
    "Joke_length_checker",
    model_client=model_client,
    system_message="You need to ensure that the length of the joke is under 200 characters in length. You can calculate the length of the joke using the 'calculate_text_length' tool. As to reduce the length of the character length if more",
    tools=[length_tool],
)


# Define a termination condition that stops the task if the critic approves.
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = text_mention_termination | max_messages_termination

team = SelectorGroupChat(
    [planning_agent, Joke_writer, Joke_reviewer, Joke_length_checker],
    model_client=model_client,
    termination_condition=termination,
)



# Streamlit input for task
task = st.text_input("Enter a task for the agents:", "write a joke")

# Button to start the conversation
if st.button("Run Conversation"):
    st.write("Running the conversation...")
    
    # Placeholder for conversation messages
    conversation_placeholder = st.empty()

    # Define a mapping of agent names to avatars
    agent_avatars = {
        "PlanningAgent": "🧠",
        "Joke_writer": "✍️",
        "Joke_reviewer": "🔍",
        "Joke_length_checker": "📏",
        "Unknown": "❓"
    }
    
    
    # Define a function to run the conversation
    async def run_conversation():
        messages = []
        try:
            # Stream messages from the team
            async for message in team.run_stream(task=task):
                # Check if the message has a 'content' attribute
                if hasattr(message, "content"):
                    # Extract sender (agent name) and content
                    sender = getattr(message, "source", "Unknown")  # Use 'source' or fallback to 'Unknown'
                    content = message.content

                    # Get the avatar for the sender
                    avatar = agent_avatars.get(sender, "❓")

                    # Append and display messages in real-time
                    messages.append(f"{avatar} **{sender}**: {content}")
                else:
                    # Handle cases where the message does not have 'content'
                    messages.append("⚠️ **System**: Received an unexpected message type.")

                # Update the conversation in real-time
                conversation_placeholder.markdown("\n\n".join(messages))  # Use markdown for better formatting
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Use asyncio to run the async function

    asyncio.run(run_conversation())

   