from __future__ import annotations

from langchain.llms.base import BaseLLM
from langchain.schema.runnable import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from pyteam.fleet_retrievers import FleetContextRetrieverMachine


class FleetBackedSpecialist:
    retriever_machine: FleetContextRetrieverMachine
    prompt: ChatPromptTemplate
    specialist: Runnable

    _system_message_template = (
        "You are a great software engineer who is very familiar with Python. "
        "Given a user question or request about a new Python library "
        "called `{library}` and parts of the `{library}` documentation, "
        "answer the question or generate the requested code. "
        "Your answers must be accurate, should include code whenever possible, "
        "and should not assume anything about `{library}` which is not "
        "explicitly stated in the `{library}` documentation. "
        "If the required information is not available, just say so.\n\n"
        "`{library}` Documentation\n"
        "------------------\n\n"
        "{context}"
    )

    _prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", _system_message_template),
            ("human", "{question}"),
        ],
    )

    def __init__(self, retriever_machine: FleetContextRetrieverMachine, llm: BaseLLM):
        self.retriever_machine = retriever_machine
        self.prompt = self._prompt_template.partial(
            library=retriever_machine.library_name,
        )
        self.specialist = (
            {
                "question": RunnablePassthrough(),
                "context": self.retriever_machine.parent_retriever
                | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            }
            | self.prompt
            | llm
            | StrOutputParser()
        )
