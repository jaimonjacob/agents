from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
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

 
# Load environment variables  
load_dotenv()  
  
# Sidebar selector to choose model  
model_choice = st.sidebar.selectbox(  
    "Select your model:",  
    ["Azure", "Gemini", "Ollama"]  
)  
  
# Initialize model_client based on selection  
if model_choice == "Azure":  
    token_provider = AzureTokenProvider(  
        DefaultAzureCredential(),  
        "https://cognitiveservices.azure.com/.default",  
    )  
    model_client = AzureOpenAIChatCompletionClient(  
          azure_deployment=os.getenv("OPENAI_API_MODEL"),
        model=os.getenv("OPENAI_API_MODEL"),  
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
        api_key=os.getenv("AZURE_OPENAI_KEY")  
    )  
elif model_choice == "Gemini":  
    model_client = OpenAIChatCompletionClient(  
        model="gemini-2.0-flash",  
        api_key=os.getenv("GEMINI_API_KEY"),  
        api_type="gemini",  
    )  
elif model_choice == "Ollama":  
    model_client = OllamaChatCompletionClient(  
        model="llama3.2:1b",  
    )  
  
st.sidebar.write(f"Current model: **{model_choice}**")  



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
    2. <agent>: <task>
    etc...

    After assigning tasks, wait for responses from the agents and ensure all subtasks are completed. After all tasks are complete, summarize the findings and end with "TERMINATE".
    """,
)

Joke_writer = AssistantAgent(
    "Joke_writer",
    model_client=model_client,
    system_message="You are a helpful AI assistant which writes the joke and edits it appropriately"    
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


selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""


# Define a termination condition that stops the task if the critic approves.
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = text_mention_termination | max_messages_termination

team = SelectorGroupChat(
    [planning_agent, Joke_writer, Joke_reviewer, Joke_length_checker],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
)



# Streamlit input for task  
task = st.text_input("Enter a task for the agents:", "write a joke")  

# Display agent names and icons in the sidebar  
st.sidebar.markdown("### Agents and Their Roles")  
agent_roles = {  
    "PlanningAgent": ("🧠", "Responsible for planning tasks."),  
    "Joke_writer": ("✍️", "Writes and edits jokes."),  
    "Joke_reviewer": ("🔍", "Reviews jokes for appropriateness."),  
    "Joke_length_checker": ("📏", "Ensures jokes are under 200 characters."),  
}  
  
for agent, (icon, role) in agent_roles.items():  
    st.sidebar.markdown(f"{icon} **{agent}**: {role}")   

# Button to start the conversation  
if st.button("Run Conversation"):  
    st.write("Running the conversation...")  
  
    # Define agent avatars  
    agent_avatars = {  
        "PlanningAgent": "🧠",  
        "Joke_writer": "✍️",  
        "Joke_reviewer": "🔍",  
        "Joke_length_checker": "📏",  
        "System": "⚙️",  
        "Unknown": "❓"  
    }  
  
    async def run_conversation():  
        terminated = False  
        try:  
            async for message in team.run_stream(task=task):  
                print(message)  
    
                sender = getattr(message, "source", "Unknown")  
    
                # Handle regular text content  
                if hasattr(message, "content") and message.content:  
                    content = message.content  
                    if isinstance(content, list):  
                        content = "\n\n".join([str(item).strip() for item in content])  
                    elif isinstance(content, str):  
                        content = content.strip()  
                    else:  
                        content = str(content).strip()  
    
                    # Check for termination  
                    if "TERMINATE" in content:  
                        terminated = True  
    
                    with st.chat_message(sender, avatar=agent_avatars.get(sender, "❓")):  
                        st.write(content)  
    
                # Handle tool calls  
                elif hasattr(message, "tool_calls") and message.tool_calls:  
                    with st.chat_message(sender, avatar=agent_avatars.get(sender, "❓")):  
                        for tool_call in message.tool_calls:  
                            function_name = tool_call.function.name  
                            function_args = tool_call.function.arguments  
                            st.markdown(f"Calling function `{function_name}` with arguments:")  
                            st.code(function_args, language="json")  
    
                # Explicitly handle TaskResult messages  
                elif message.__class__.__name__ == "TaskResult":  
                    terminated = True  
                    with st.chat_message("System", avatar=agent_avatars.get("System")):  
                        st.success("✅ Task Completed Successfully!")  
                        st.markdown(f"**Stop reason:** {message.stop_reason}")  
                        st.markdown("**Conversation Summary:**")  
                        for msg in message.messages:  
                            msg_source = getattr(msg, "source", "Unknown")  
                            msg_content = getattr(msg, "content", "")  
                            if isinstance(msg_content, list):  
                                msg_content = "\n\n".join([str(item).strip() for item in msg_content])  
                            st.markdown(f"- **{msg_source}**: {msg_content}")  
    
                else:  
                    # Fallback for any other unexpected message types  
                    with st.chat_message("System", avatar=agent_avatars.get("System")):  
                        st.warning(f"⚠️ Received an unexpected message type: {message.type}")  
    
            if terminated:  
                with st.chat_message("System", avatar=agent_avatars.get("System")):  
                    st.success("✅ Conversation fully completed.")  
    
        except Exception as e:  
            st.error(f"An error occurred: {e}")  
    # Execute the async conversation function  
    asyncio.run(run_conversation())  