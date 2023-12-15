from __future__ import annotations

from typing import Optional

import pandas as pd
from langchain.llms.base import BaseLLM
from langchain.schema.document import Document
from langchain.schema.runnable import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from pyteam.fleet_retrievers import MultiVectorFleetRetriever


class FleetBackedSpecialist:
    retriever: MultiVectorFleetRetriever
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

    @staticmethod
    def _join_docs(docs: list[Document], sep: str = "\n\n") -> str:
        return sep.join(d.page_content for d in docs)

    def __init__(
        self,
        library_name: str,
        retriever: MultiVectorFleetRetriever,
        llm: BaseLLM,
    ):
        self.retriever = retriever
        self.prompt = self._prompt_template.partial(
            library=library_name,
        )
        self.specialist = (
            {
                "question": RunnablePassthrough(),
                "context": self.retriever | (lambda docs: self._join_docs(docs)),
            }
            | self.prompt
            | llm
            | StrOutputParser()
        )

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        library_name: str,
        llm: BaseLLM,
        **kwargs,
    ) -> FleetBackedSpecialist:
        retriever = MultiVectorFleetRetriever.from_df(
            df,
            library_name,
            **kwargs,
        )
        return cls(library_name, retriever, llm)

    @classmethod
    def from_library(
        cls,
        library_name: str,
        llm: BaseLLM,
        download_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> FleetBackedSpecialist:
        retriever = MultiVectorFleetRetriever.from_library(
            library_name,
            download_kwargs,
            **kwargs,
        )
        return cls(library_name, retriever, llm)

    @classmethod
    def from_parquet(
        cls,
        parquet_path,
        llm: BaseLLM,
        **kwargs,
    ) -> FleetBackedSpecialist:
        retriever = MultiVectorFleetRetriever.from_parquet(
            parquet_path,
            **kwargs,
        )
        library_name = MultiVectorFleetRetriever.get_library_name_from_filename(
            parquet_path,
        )
        return cls(library_name, retriever, llm)
