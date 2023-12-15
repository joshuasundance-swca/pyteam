from typing import Optional

from langchain.llms.base import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from pyteam import FleetContextRetrieverMachine

system_message = (
    "You are a great software engineer who is very familiar with Python. "
    "Given a user question or request about a new Python library "
    "called {library} and parts of the {library} documentation, "
    "answer the question or generate the requested code. "
    "Your answers must be accurate, should include code whenever possible, "
    "and should not assume anything about {library} which is not "
    "explicitly stated in the {library} documentation. "
    "If the required information is not available, just say so.\n\n"
    "{library} Documentation\n"
    "------------------\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "{question}"),
    ],
)


def get_specialist(library: str, llm: BaseLLM, download_kwargs: Optional[dict] = None):
    return (
        {
            "question": RunnablePassthrough(),
            "context": FleetContextRetrieverMachine.from_library(
                library,
                download_kwargs,
            ).parent_retriever
            | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        }
        | prompt.partial(library=library)
        | llm
        | StrOutputParser()
    )
