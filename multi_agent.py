import os
import autogen

config_list = autogen.config_list_from_json(
    env_or_file="OAI_CONFIG_LIST.json"
)

llm_config = {
     "config_list":config_list,
     "cache_seed": 41,
     "temperature": 0,
     "timeout": 120
}

assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config  
)

user_proxy = autogen.UserProxyAgent(
    name="admin",
    system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin",
    code_execution_config={
        "work_dir": "code",
        "user_docker": False
    },
    human_input_mode="TERMINATE"
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=llm_config,
    system_message=""" Enginner: You follow an approved plan. Make sure you save code to desk.
    You write python/shell code to solve tasks. Wrap the code in a block that specifies the script type and the name of the file to save to the desk """

)

scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=llm_config,
    system_message=""" Scientist. You follow an approved plan. You are able to categorize papers after seeing their 
    abstracts printed. You don't write code."""

)

planner = autogen.AssistantAgent(
    name="Planner",
    system_message="""Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
    llm_config=llm_config,
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the "
                   "plan includes adding verifiable info such as source URL.",
    llm_config=llm_config,
)

group_chat = autogen.GroupChat(
    agents=[user_proxy, engineer, scientist, planner, critic], messages=[], max_round=12
)

manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message="""
Find papers on LLM applications from arxiv in the last week, create a markdown table of different domains.
""")