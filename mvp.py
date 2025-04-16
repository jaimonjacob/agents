from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
import asyncio
import os
from dotenv import load_dotenv
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import Tool  # Import the Tool class
from bs4 import BeautifulSoup
import re
from autogen_agentchat.ui import Console

 
# Load environment variables  
load_dotenv()  
  

# token_provider = AzureTokenProvider(  
#     DefaultAzureCredential(),  
#     "https://cognitiveservices.azure.com/.default",  
# )  
# model_client = AzureOpenAIChatCompletionClient(  
#     azure_deployment=os.getenv("OPENAI_API_MODEL"),
#     model=os.getenv("OPENAI_API_MODEL"),  
#     api_version=os.getenv("OPENAI_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
#     api_key=os.getenv("AZURE_OPENAI_KEY")  
# )    

model_client = OpenAIChatCompletionClient(  
        model="gemini-2.0-flash",  
        api_key=os.getenv("GEMINI_API_KEY"),  
        api_type="gemini",  
    )  

print(model_client)

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
url = "https://www.codecademy.com/article/accessibility-on-the-platform"  

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

async def main():   
    await Console(team.run_stream(task=task))    
asyncio.run(main())