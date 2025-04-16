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
from bs4 import BeautifulSoup
import re
from streamlit_avatar import avatar


# Streamlit UI
st.title("AutoGen Multi-Agent System")
st.write("This app visualizes the conversation between agents working collaboratively to create a FAQ from a URL.")

 
# Load environment variables  
load_dotenv()  
  
# Sidebar selector to choose model  
model_choice = st.sidebar.selectbox(  
    "Select your model:",  
    ["Gemini", "Azure", "Ollama"] ,
     
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

def fetch_url_text_tool(url: str) -> str:from bs4 import BeautifulSoup
import re

def clean_text(text: str) -> str:
    """Remove unnecessary characters and whitespace."""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def fetch_url_text_tool(url: str) -> str:
    """Fetch the main text content from a webpage and clean it."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        if docs and docs[0].page_content.strip():
            # Use BeautifulSoup to strip out HTML tags
            soup = BeautifulSoup(docs[0].page_content, "html.parser")
            main_text = soup.get_text()
            
            return clean_text(main_text)
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
#Display agent names and icons in the sidebar  
# st.sidebar.markdown("### Agents and their Roles")  
# agent_roles = {  
#     "ProjectPlanner": ("👨‍💼", "Responsible for planning tasks."),  
#     "CrawlerAgent": ("💂‍♀️", "Extracts useful text from a given URL."),  
#     "IndexerAgent": ("👩‍✈️", "Organizes content into tagged categories."),  
#     "FAQGeneratorAgent": ("🦸‍♂️", "Generates helpful Q&A pairs for each category."),  
#     "VerifierAgent": ("👩‍🚒", "Polishes, deduplicates, and validates the final Q&A content."),  
#     "System": ("⚙", "Handles system-level operations."),  
#     "Unknown": ("❓", "Represents an unidentified agent.")  
# }
# for agent, (icon, role) in agent_roles.items():  
#     st.sidebar.markdown(f"{icon} **{agent}**: {role}")   

agent_avatars = [
        {
            "url": "icons/project_planner_avatar.png",
            "size": 40,
            "title": "ProjectPlanner",
            "caption": "Responsible for planning tasks",
            "key": "project_planner",
        },
        {
            "url": "icons/crawler_avatar.png",
            "size": 40,
            "title": "CrawlerAgent",
            "caption": "Extracts useful text from a given URL",
            "key": "crawler_agent",
        },
        {
            "url": "icons/indexer_avatar.png",
            "size": 40,
            "title": "IndexerAgent",
            "caption": "Organizes content into tagged categories",
            "key": "indexer_agent",
        },
        {
            "url": "icons/faq_generator_avatar.png",
            "size": 40,
            "title": "FAQGeneratorAgent",
            "caption": "Generates helpful Q&A pairs for each category",
            "key": "faq_generator_agent",
        },
        {
            "url": "icons/verifier_avatar.png",
            "size": 40,
            "title": "VerifierAgent",
            "caption": "Polishes and validates the final Q&A content",
            "key": "verifier_agent",
        },
        {
            "url": "icons/system_avatar.png",
            "size": 40,
            "title": "System",
            "caption": "Handles system-level operations",
            "key": "system",
        },
          {
            "url": "icons/unknown_avatar.png",
            "size": 40,
            "title": "Unknown",
            "caption": "Any system errors",
            "key": "Unknown",
        },
    ]

st.sidebar.markdown("### Agents and their Roles")
for agent in agent_avatars:
    # Display the avatar image
    st.sidebar.image(agent["url"], width=40)
    # Display the title and caption
    st.sidebar.markdown(f"**{agent['title']}**: {agent['caption']}") 

# Button to start the conversation  
if st.button("Run task"):  
    st.write("Running the task...")  
  
    # # Define agent avatars  
    # agent_avatars = {  
    #     "ProjectPlanner": "👨‍💼",  
    #     "CrawlerAgent": "💂‍♀️",  
    #     "IndexerAgent": "👩‍✈️",  
    #     "FAQGeneratorAgent": "🦸‍♂️",  
    #     "VerifierAgent": "👩‍🚒",
    #     "System": "⚙",  
    #     "Unknown": "❓"  
    # }  
  
    
    
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
    
                    # Display messages with avatars
                    with st.chat_message(sender):
                        avatar_url = next((a["url"] for a in agent_avatars if a["title"] == sender), None)
                        if avatar_url:
                            st.image(avatar_url, width=40)
                        st.write(content)
    
                # Handle tool calls  
                elif hasattr(message, "tool_calls") and message.tool_calls:  
                    with st.chat_message(sender):  
                        avatar_url = next((a["url"] for a in agent_avatars if a["title"] == sender), None)
                        if avatar_url:
                            st.image(avatar_url, width=40)
                        for tool_call in message.tool_calls:  
                            function_name = tool_call.function.name  
                            function_args = tool_call.function.arguments  
                            st.markdown(f"Calling function `{function_name}` with arguments:")  
                            st.code(function_args, language="json")  
    
                # Explicitly handle TaskResult messages  
                elif message.__class__.__name__ == "TaskResult":  
                    terminated = True  
                    with st.chat_message("System"):  
                        avatar_url = next((a["url"] for a in agent_avatars if a["title"] == "System"), None)
                        if avatar_url:
                            st.image(avatar_url, width=40)
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
                    with st.chat_message("System"):  
                        avatar_url = next((a["url"] for a in agent_avatars if a["title"] == "System"), None)
                        if avatar_url:
                            st.image(avatar_url, width=40)
                        st.warning(f"⚠️ Received an unexpected message type: {message.type}")  
    
            if terminated:  
                with st.chat_message("System"):  
                    avatar_url = next((a["url"] for a in agent_avatars if a["title"] == "System"), None)
                    if avatar_url:
                        st.image(avatar_url, width=40)
                    st.success("✅ Conversation fully completed.")  
    
        except Exception as e:  
            st.error(f"An error occurred: {e}")  
    # Execute the async conversation function  
    asyncio.run(run_conversation())