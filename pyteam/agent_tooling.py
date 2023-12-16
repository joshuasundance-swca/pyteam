from typing import Optional

from langchain.agents import AgentExecutor
from langchain.agents import AgentType, initialize_agent
from langchain.agents.tools import BaseTool
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.tools import StructuredTool

from pyteam.fleet_specialists import FleetBackedSpecialist

_system_message_template = (
    "You are responsible for delivering a Python project. "
    "Your team is composed of expert programmers.\n"
    "You know a lot about what different libraries are used for, "
    "but you don't really know how to use them. :/\n"
    "That's okay though! Your team is a bunch of rockstars.\n"
    "Delegate tasks based on the instructions and requirements below, then synthesize the results.\n\n"
    "Be sure to lean on your team of specialists, and summon new help if you need to."
    "Return the code only once you're 100% sure it's ready. Don't be lazy, and don't rush. "
    "I'd hate to see you and your team get fired. <_<\n\n"
    "If you do a good job, we'll all get a huge bonus!\n"
    "So delegate and synthesize carefully. Thank you!"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system_message_template),
        ("human", "{question}"),
        (
            "human",
            "Remember to use the `ask` tools to communicate with your team. There is no 'I' in 'team'.",
        ),
    ],
)


class SpecialistRequest(BaseModel):
    library_name: str
    request: str


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
            ).specialist
            self.specialists[library_name] = specialist

    def _ask_specialist(self, library_name: str, request: str) -> str:
        try:
            self._get_specialist(library_name)
            ch = self.specialists[library_name] | StrOutputParser()
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
            return (ch | StrOutputParser()).invoke(dict(question=request))

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
        ]

        if initial_libraries:
            for library in initial_libraries:
                self._get_specialist(library)

        self.agent = self.get_agent_executor()

    def __call__(self, instructions: str) -> dict[str, str]:
        return self.agent(instructions)


# # llm = ...
# a = SpecialistBackedAgent(llm)  #, ['langchain', 'pandas'])
#
# instructions = """
# Write a script that does the following:
#
# pandas (have your pandas assistant write this)
# 1. reads `local.csv`
# 2. sort by 'id' and 'name'
#
# langchain (have your langchain assistant write this)
# 3. make a faiss vectorstore from the 'text' column. put 'id' and 'name' in the metadata
#
#
# develop the script step by step with the help of relevant experts. it may be tricky.
#
# if you do this right, I'll give you $200! :)
# """.strip()
# result = a.agent(instructions)
