from autogen_ext.auth.azure import AzureTokenProvider
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from azure.identity import DefaultAzureCredential
import json
from pathlib import Path
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
import asyncio
import os
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
from autogen_ext.tools.langchain import LangChainToolAdapter
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain.tools import Tool  # Import the Tool class


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

def fetch_url_text_tool(url: str) -> str:
    """Fetch the main text content from a webpage."""    
    loader = WebBaseLoader(url)
    docs = loader.load()
    print (docs[0].page_content if docs else "No content found at the URL.")
    return docs[0].page_content if docs else "No content found at the URL."

# Wrap the function in a LangChain Tool
fetch_url_text_tool_wrapped = Tool(
    name="fetch_url_text",
    func=fetch_url_text_tool,
    description="Fetch the main text content from a webpage."
)


fetch_url_text = LangChainToolAdapter(fetch_url_text_tool_wrapped)


def load_input_texts(folder_path="inputs"):
    texts = []
    for file in Path(folder_path).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return "\n\n".join(texts)

# raw_text = load_input_texts()



project_planner = AssistantAgent(
    name="ProjectPlanner",
    model_client=model_client,
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    system_message="""
    You are a planning agent.
    You only plan and delegate tasks - you do not execute them yourself. You can engage team members multiple times so that a perfect Joke is provided.
    Your team members are CrawlerAgent, indexer, qa_gen, and verifier.    
    After assigning tasks, wait for responses from the agents and ensure all subtasks are completed. After all tasks are complete, summarize the findings and end with "TERMINATE". Do not mention "TERMINATE" before that.
    """           
)


crawler = AssistantAgent(
    name="CrawlerAgent",
    model_client=model_client,
    system_message="You are responsible for extracting useful text from a given URL using the fetch_url_text tool.",
    tools=[fetch_url_text]  # 👈 tool added here
)

indexer = AssistantAgent(
    name="IndexerAgent",
    model_client=model_client,
    system_message="You organize content into tagged categories and prepare it for Q&A generation."
)

qa_gen = AssistantAgent(
    name="QAGeneratorAgent",
    model_client=model_client,
    system_message="You generate helpful Q&A pairs for each category."
)

verifier = AssistantAgent(
    name="VerifierAgent",
    model_client=model_client,
    system_message="You polish, deduplicate, and validate the final Q&A content."
)



# Define a termination condition that stops the task if the critic approves.
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=10)
termination = text_mention_termination | max_messages_termination

team = SelectorGroupChat(
    [project_planner, crawler, indexer, qa_gen, verifier],    
    termination_condition=termination,
    model_client=model_client,
    allow_repeated_speaker=True,  
)

async def main():
    os.makedirs("output", exist_ok=True)

    task = """
    Please build a structured knowledge base from the following URL:

    https://en.wikipedia.org/wiki/How_Brown_Saw_the_Baseball_Game

    Use the fetch_url_text tool to retrieve the content, then analyze it.

    Break down into categories and generate Q&A pairs like:
    [
    {
        "category": "Overview",
        "questions": [
        {"question": "What is cybersecurity?", "answer": "..."}
        ]
    },
    ...
    ]
.
    """ 
        # Await the team.run() coroutine and pass the result to Console
    await Console(
        team.run_stream(task=task)
    )  # Stream the messages to the console.
    # await team.run(task="Teach me oop with Python")
    # print("\n🔧 Final output:\n", result.output)
    #  # Save output
    # os.makedirs("output", exist_ok=True)
    # with open("output/knowledge_base.json", "w", encoding="utf-8") as f:
    #     json.dump(result.output, f, indent=2)

    # print("✅ Knowledge base saved to output/knowledge_base.json")

asyncio.run(main())
