from typing import Optional

from langchain.agents import AgentExecutor
from langchain.agents import AgentType, initialize_agent
from langchain.agents.tools import Tool
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import Runnable
from langchain.tools import tool

from pyteam.fleet_retrievers import MultiVectorFleetRetriever
from pyteam.fleet_specialists import FleetBackedSpecialist


def request_specialist(
    library_name: str,
    llm: BaseLLM,
    download_kwargs: Optional[dict] = None,
) -> Runnable:
    retriever = MultiVectorFleetRetriever.from_library(library_name, download_kwargs)
    return FleetBackedSpecialist(library_name, retriever, llm).specialist


llm = ...
download_kwargs = dict(cache_dir="fleet-cache")

tools = []


@tool("add_specialist", "Add a specialist for a specific Python library")
def add_specialist(library_name: str) -> str:
    global tools
    retriever = MultiVectorFleetRetriever.from_library(library_name, download_kwargs)
    specialist = FleetBackedSpecialist(library_name, retriever, llm).specialist
    specialist_tool = Tool.from_function(
        specialist.invoke,
        name=f"{library_name} specialist",
        description=f"Specialist for {library_name}",
    )
    tools.append(specialist_tool)
    return specialist_tool.name


class ToolGrabber:
    tools: list[Tool]
    agent: AgentExecutor

    def add_tool(self, tool_name: str) -> str:
        tool = Tool.from_function(
            lambda s: "WOOF WOOF WOOF WOOF",
            name=tool_name,
            description=f"Tool for {tool_name}",
            return_direct=True,
        )
        self.tools.append(tool)
        self.agent = self.get_agent_executor(llm, self.agent.memory)
        return tool.name

    def get_agent_executor(
        self,
        llm: BaseLLM,
        memory,
        agent_type: AgentType = AgentType.OPENAI_FUNCTIONS,
    ) -> AgentExecutor:
        return initialize_agent(
            self.tools,
            llm,
            agent=agent_type,
            verbose=True,
            agent_kwargs={
                "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            },
            memory=memory,
            handle_parsing_errors=True,
        )

    def __init__(self, llm: BaseLLM):
        self.tools = [
            Tool.from_function(
                self.add_tool,
                name="create_tool",
                description="Create a tool (just give it a name)",
            ),
        ]

        self.agent = self.get_agent_executor(
            llm,
            ConversationBufferMemory(memory_key="memory", return_messages=True),
        )
