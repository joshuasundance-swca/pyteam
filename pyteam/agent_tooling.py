from langchain.agents import AgentExecutor
from langchain.agents import AgentType, initialize_agent
from langchain.agents.tools import Tool
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

from pyteam.fleet_specialists import FleetBackedSpecialist


class SpecialistBackedAgent:
    tools: list[Tool]
    llm: BaseLLM
    cache_dir: str = "fleet-cache"
    agent: AgentExecutor

    def _summon_specialist(self, library_name: str) -> str:
        specialist = FleetBackedSpecialist.from_library(
            library_name,
            self.llm,
            dict(cache_dir=self.cache_dir),
        ).specialist
        tool = Tool.from_function(
            specialist.invoke,
            name=library_name,
            description=f"Get advice from a {library_name} specialist",
            return_direct=True,
        )
        self.tools.append(tool)
        self.agent = self.get_agent_executor(self.llm, self.agent.memory)
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
        self.llm = llm

        self.tools = [
            Tool.from_function(
                self._summon_specialist,
                name="summon_specialist",
                description="Summon a specialist for a Python library",
            ),
        ]

        self.agent = self.get_agent_executor(
            llm,
            ConversationBufferMemory(memory_key="memory", return_messages=True),
        )
