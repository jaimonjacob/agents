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
from dotenv import load_dotenv
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import Tool  # Import the Tool class
import streamlit as st

# Streamlit UI
st.title("AutoGen Multi-agent System")
st.write("This app visualizes the conversation between agents working collaboratively to create a FAQ from a URL.")

 
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

def fetch_url_text_tool(url: str) -> str:
    """Fetch the main text content from a webpage."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if docs and docs[0].page_content.strip():
            return docs[0].page_content
        else:
            return "Error: No content found at the provided URL."
    except Exception as e:
        return f"Error: Failed to fetch content from the URL. Details: {str(e)}"

# Wrap the function in a LangChain Tool
fetch_url_text_tool_wrapped = Tool(
    name="fetch_url_text",
    func=fetch_url_text_tool,
    description="Fetch the main text content from a webpage."
)


fetch_url_text = LangChainToolAdapter(fetch_url_text_tool_wrapped)

project_planner = AssistantAgent(
    name="ProjectPlanner",
    model_client=model_client,
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    system_message="""
    You are a planning agent.
    You only plan and delegate tasks - you do not execute them yourself. You can engage team members multiple times so that a perfect Joke is provided.
    Your team members are CrawlerAgent, indexer, FAQ_generator, and verifier.    
    After assigning tasks, wait for responses from the agents, handover tasks between agents, and ensure all subtasks are completed. After all tasks are complete, summarize the findings and end with "TERMINATE". Do not mention "TERMINATE" before that.
    """           
)


crawler = AssistantAgent(
    name="CrawlerAgent",
    model_client=model_client,
    system_message="You are responsible for extracting useful text from a given URL using the fetch_url_text tool. ",
    tools=[fetch_url_text]  # 👈 tool added here
)

indexer = AssistantAgent(
    name="IndexerAgent",
    model_client=model_client,
    system_message="You organize content into tagged categories and prepare it for Q&A generation. "
)

FAQ_generator = AssistantAgent(
    name="FAQGeneratorAgent",
    model_client=model_client,
    system_message="You generate helpful Q&A pairs for each category. "
)

verifier = AssistantAgent(
    name="VerifierAgent",
    model_client=model_client,
    system_message="You polish, deduplicate, and validate the final Q&A content. "
)



# Define a termination condition that stops the task if the critic approves.
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = text_mention_termination | max_messages_termination

team = SelectorGroupChat(
    [project_planner, crawler, indexer, FAQ_generator, verifier],    
    termination_condition=termination,
    model_client=model_client,
    allow_repeated_speaker=True,  
)



# Streamlit input for task  
url = st.text_input("Add the URL here", "https://en.wikipedia.org/wiki/How_Brown_Saw_the_Baseball_Game")  

task = f"""
    Please build a structured knowledge base from the following URL:

    {url}

    Use the fetch_url_text tool to retrieve the content, then analyze it.

    Break down into categories and generate Q&A pairs like in json format:
    [
        {{
            "category": "Overview",
            "questions": [
                {{"question": "What is cybersecurity?", "answer": "..."}}
            ]
        }},
        ...
    ]
"""
# Display agent names and icons in the sidebar  
st.sidebar.markdown("### Agents and their Roles")  
agent_roles = {  
    "ProjectPlanner": ("👨‍💼", "Responsible for planning tasks."),  
    "CrawlerAgent": ("💂‍♀️", "Extracts useful text from a given URL."),  
    "IndexerAgent": ("👩‍✈️", "Organizes content into tagged categories."),  
    "FAQGeneratorAgent": ("🦸‍♂️", "Generates helpful Q&A pairs for each category."),  
    "VerifierAgent": ("👩‍🚒", "Polishes, deduplicates, and validates the final Q&A content."),  
    "System": ("⚙", "Handles system-level operations."),  
    "Unknown": ("❓", "Represents an unidentified agent.")  
}
for agent, (icon, role) in agent_roles.items():  
    st.sidebar.markdown(f"{icon} **{agent}**: {role}")   

# Button to start the conversation  
if st.button("Run task"):  
    st.write("Running the task...")  
  
    # Define agent avatars  
    agent_avatars = {  
        "ProjectPlanner": "👨‍💼",  
        "CrawlerAgent": "💂‍♀️",  
        "IndexerAgent": "👩‍✈️",  
        "FAQGeneratorAgent": "🦸‍♂️",  
        "VerifierAgent": "👩‍🚒",
        "System": "⚙",  
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
    
                    # Check for error messages from the fetch_url_text_tool
                    if content.startswith("Error:"):
                        with st.chat_message(sender, avatar=agent_avatars.get(sender, "❓")):
                            st.error(content)
                    else:
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