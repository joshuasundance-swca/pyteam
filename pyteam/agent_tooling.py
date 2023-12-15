from typing import Optional

from langchain.llms.base import BaseLLM
from langchain.schema.runnable import Runnable

from pyteam.fleet_retrievers import MultiVectorFleetRetriever
from pyteam.fleet_specialists import FleetBackedSpecialist


def request_specialist(
    library_name: str,
    llm: BaseLLM,
    download_kwargs: Optional[dict] = None,
) -> Runnable:
    retriever = MultiVectorFleetRetriever.from_library(library_name, download_kwargs)
    return FleetBackedSpecialist(library_name, retriever, llm).specialist
