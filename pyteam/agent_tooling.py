from typing import Optional

from langchain.agents import AgentExecutor
from langchain.agents import AgentType, initialize_agent
from langchain.agents.tools import Tool
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate

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


class SpecialistBackedAgent:
    tools: list[Tool]
    llm: BaseLLM
    cache_dir: str = "fleet-cache"
    agent: AgentExecutor

    def _summon_specialist(self, library_name: str) -> str:
        tool_name = f"ask-{library_name}"
        try:
            specialist = FleetBackedSpecialist.from_library(
                library_name,
                self.llm,
                dict(cache_dir=self.cache_dir),
            ).specialist
        except ValueError:
            return (
                f"Sorry, no specialist is available for {library_name}. "
                "Double-check the library name and try again, "
                "or try a different library."
            )
        msg = (
            f"Success! You may now use the `{tool_name}` tool "
            f"to ask the {library_name} specialist for help."
        )
        tool = Tool.from_function(
            specialist.invoke,
            name=tool_name,
            description=f"Ask the {library_name} specialist for help.",
        )
        self.tools.append(tool)
        self.agent = self.get_agent_executor(self.llm, self.agent.memory)
        return msg

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
            prompt=prompt,
            # return_intermediate_steps=True,
        )

    def __init__(self, llm: BaseLLM, initial_libraries: Optional[list[str]] = None):
        self.llm = llm

        self.tools = [
            Tool.from_function(
                lambda x: x,
                name="submit-final-code",
                description="This is the final step. "
                "Submit your FINAL, COMPLETE, SYNTHESIZED "
                "code to the user who requested it. "
                "Never submit code without getting input from relevant team members.",
                return_direct=True,
            ),
            Tool.from_function(
                self._summon_specialist,
                name="hire-specialist",
                description="Hire a specialist for a specific Python library.",
            ),
        ]

        self.agent = self.get_agent_executor(
            llm,
            ConversationBufferMemory(memory_key="memory", return_messages=True),
        )

        if initial_libraries:
            for library in initial_libraries:
                self._summon_specialist(library)


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
# if you do this right, I'll give you $200! :)
# """.strip()
# result = a.agent(instructions)
