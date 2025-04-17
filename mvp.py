import os
import re
import asyncio
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from langchain.tools import Tool
from langchain_community.document_loaders import WebBaseLoader
from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.tools.langchain import LangChainToolAdapter

# Load environment variables
load_dotenv()

# Initialize the model client
model_client = AzureOpenAIChatCompletionClient(  
    azure_deployment=os.getenv("AZURE_OPENAI_API_MODEL"),
    model=os.getenv("AZURE_OPENAI_API_MODEL"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),      
    api_key=os.getenv("AZURE_OPENAI_KEY") 
)  

# Creating a tool
def clean_text(text: str) -> str:
    """Remove unnecessary characters and whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def fetch_url_text_tool(url: str) -> str:
    """Fetch and clean the main text content from a webpage."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if docs and docs[0].page_content.strip():
            soup = BeautifulSoup(docs[0].page_content, "html.parser")
            return clean_text(soup.get_text())
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

# Define agents
project_planner = AssistantAgent(
    name="ProjectPlanner",
    model_client=model_client,
    description="An agent for planning tasks.",
    system_message=
        """You are a planning agent. You only plan and delegate tasks - you do not execute them yourself.
        You can engage team members multiple times to ensure a perfect Joke is provided. 
        Your team members are CrawlerAgent, IndexerAgent, FAQGeneratorAgent, and VerifierAgent. 
        After assigning tasks, wait for responses from the agents, handover tasks between agents if needed, and ensure all subtasks are completed. 
        "After all tasks are complete, summarize the findings and end with 'TERMINATE'. Do not mention 'TERMINATE' before that."""
    
)

crawler = AssistantAgent(
    name="CrawlerAgent",
    model_client=model_client,
    system_message="You are responsible for extracting useful text from a given URL using the fetch_url_text tool.",
    tools=[fetch_url_text]
)

indexer = AssistantAgent(
    name="IndexerAgent",
    model_client=model_client,
    system_message="You organize content into tagged categories and prepare it for Q&A generation."
)

FAQ_generator = AssistantAgent(
    name="FAQGeneratorAgent",
    model_client=model_client,
    system_message="You generate helpful Q&A pairs for each category."
)

verifier = AssistantAgent(
    name="VerifierAgent",
    model_client=model_client,
    system_message="You polish, deduplicate, and validate the final Q&A content."
)

# Define termination conditions
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=10)

# Define the team
team = SelectorGroupChat(
    [project_planner, crawler, indexer, FAQ_generator, verifier],
    termination_condition=termination,
    model_client=model_client,
    allow_repeated_speaker=True,
)

# Define the task
url = "https://plus.maths.org/content/ridiculously-brief-introduction-quantum-mechanics"
task = f"""
    Please build a structured knowledge base from the following URL:

    {url}

    Use the fetch_url_text tool to retrieve the content, then analyze it.

    Break down into categories and generate Q&A pairs like in JSON format:
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

# Main function
async def main():
    await Console(team.run_stream(task=task))

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
