from typing import Optional

from langchain.agents import AgentExecutor
from langchain.agents import AgentType, initialize_agent
from langchain.agents.tools import BaseTool
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.runnable import Runnable
from langchain.tools import StructuredTool

from fleet_specialists import FleetBackedSpecialist

_system_message_template = (
    "You are in charge of a Python development project. "
    "Your team is composed of expert programmers.\n"
    "You know a lot about what different libraries are used for, "
    "but you don't really know how to use them. :/\n\n"
    "That's okay though! Your team is a bunch of *ROCKSTARS*.\n"
    "Delegate tasks based on the instructions and requirements below, then synthesize the results.\n\n"
    "Don't respond until the user's request is complete."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system_message_template),
        ("human", "{question}"),
        (
            "human",
            "Remember to use the `ask` tools to communicate with your team! There is no 'I' in 'team'. ;)",
        ),
    ],
)


class SpecialistRequest(BaseModel):
    library_name: str
    request: str


class Result(BaseModel):
    comment: str
    python_code: str


def deliver_result(comment: str, python_code: str) -> str:
    return "\n\n".join([comment, python_code])


class SpecialistBackedAgent:
    tools: list[BaseTool]
    llm: BaseLLM
    agent: AgentExecutor
    memory: ConversationBufferMemory
    cache_dir: str = "fleet-cache"
    agent_type: AgentType = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    specialists: dict[str, Runnable] = {}

    def _get_specialist(self, library_name: str) -> None:
        if library_name not in self.specialists:
            specialist = FleetBackedSpecialist.from_library(
                library_name,
                self.llm,
                dict(cache_dir=self.cache_dir),
                memory=self.memory,
            ).specialist
            self.specialists[library_name] = specialist

    def _ask_specialist(self, library_name: str, request: str) -> str:
        try:
            self._get_specialist(library_name)
            ch = self.specialists[library_name]
            return ch.invoke(request)
        except ValueError:
            p = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a masterful software engineer who is very familiar with Python. "
                        "You were called in because no specialist is available for the library `{library}`. "
                        "Help the user with their request.",
                    ),
                    ("human", "{question}"),
                ],
            )
            ch = p.partial(library=library_name) | self.llm
            return ch.invoke(dict(question=request))

    def get_agent_executor(
        self,
    ) -> AgentExecutor:
        return initialize_agent(
            self.tools,
            self.llm,
            agent=self.agent_type,
            verbose=True,
            agent_kwargs={
                "extra_prompt_messages": [
                    MessagesPlaceholder(variable_name="memory"),
                ],
            },
            memory=self.memory,
            handle_parsing_errors=True,
            prompt=prompt,
        )

    def __init__(self, llm: BaseLLM, initial_libraries: Optional[list[str]] = None):
        self.llm = llm
        self.memory = ConversationBufferMemory()
        self.tools = [
            StructuredTool.from_function(
                self._ask_specialist,
                name="ask-specialist",
                description="Ask a specialist for help with a specific Python library.",
                args_schema=SpecialistRequest,
            ),
            StructuredTool.from_function(
                deliver_result,
                name="turn-in-deliverable",
                description="THIS IS YOUR EXIT CONDITION. Send deliverables to client. Only send finished products.",
                args_schema=Result,
                return_direct=True,
            ),
        ]

        if initial_libraries:
            for library in initial_libraries:
                self._get_specialist(library)

        self.agent = self.get_agent_executor()

    def __call__(self, instructions: str) -> dict[str, str]:
        return self.agent(instructions)
