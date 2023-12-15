from __future__ import annotations

import re
import warnings
from typing import Optional

import pandas as pd
from context import download_embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.schema.storage import BaseStore
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.storage.in_memory import InMemoryStore
from langchain.vectorstores.faiss import FAISS


class FleetContextRetrieverMachine:
    """A class to create retrievers from `fleet-context` embeddings."""

    library_name: str
    vectorstore: FAISS
    vecstore_retriever: VectorStoreRetriever
    parent_retriever: MultiVectorRetriever

    @staticmethod
    def _join_metadata(df: pd.DataFrame) -> pd.DataFrame:
        """Join metadata columns to df."""
        return df.join(
            df["metadata"].apply(pd.Series),
            lsuffix="_orig",
            rsuffix="_md",
        )

    @staticmethod
    def _df_to_parent_docs(joined_df: pd.DataFrame, sep: str = "\n") -> list[Document]:
        """Convert joined df to parent docs."""
        return (
            joined_df[["parent", "title", "text", "type", "url", "section_index"]]
            .rename(columns={"parent": "id"})
            .sort_values(["id", "section_index"])
            .groupby("id")
            .apply(
                lambda chunk: Document(
                    page_content=chunk.iloc[0]["title"]
                    + "\n"
                    + chunk["text"].str.cat(sep=sep),
                    metadata=chunk.iloc[0][["title", "type", "url", "id"]].to_dict(),
                ),
            )
            .tolist()
        )

    @staticmethod
    def _get_vectorstore(joined_df: pd.DataFrame, **kwargs) -> FAISS:
        """Get FAISS vectorstore from joined df."""
        return FAISS.from_embeddings(
            joined_df[["text", "dense_embeddings"]].values,
            OpenAIEmbeddings(model="text-embedding-ada-002"),
            metadatas=joined_df["metadata"].tolist(),
            **kwargs,
        )

    @classmethod
    def _get_parent_retriever(
        cls,
        joined_df: pd.DataFrame,
        vectorstore: FAISS,
        docstore: Optional[BaseStore] = None,
        parent_doc_sep: str = "\n",
        **kwargs,
    ) -> MultiVectorRetriever:
        """Get MultiVectorRetriever from joined df."""
        docstore = docstore or InMemoryStore()
        parent_docs = cls._df_to_parent_docs(joined_df, sep=parent_doc_sep)
        docstore.mset([(doc.metadata["id"], doc) for doc in parent_docs])
        return MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="parent",
            **kwargs,
        )

    def __init__(
        self,
        df: pd.DataFrame,
        library_name: str,
        docstore: Optional[BaseStore] = None,
        parent_doc_sep: str = "\n",
        vectorstore_kwargs: Optional[dict] = None,
        vecstore_retriever_kwargs: Optional[dict] = None,
        parent_retriever_kwargs: Optional[dict] = None,
    ):
        self.library_name = library_name

        joined_df = self._join_metadata(df)

        vectorstore_kwargs = vectorstore_kwargs or {}
        vecstore_retriever_kwargs = vecstore_retriever_kwargs or {}
        parent_retriever_kwargs = parent_retriever_kwargs or {}

        self.vectorstore = self._get_vectorstore(joined_df, **vectorstore_kwargs)
        self.vecstore_retriever = self.vectorstore.as_retriever(
            **vecstore_retriever_kwargs,
        )
        self.parent_retriever = self._get_parent_retriever(
            joined_df,
            self.vectorstore,
            docstore,
            parent_doc_sep=parent_doc_sep,
            **parent_retriever_kwargs,
        )

    def retrievers(self) -> tuple[VectorStoreRetriever, MultiVectorRetriever]:
        """Return retrievers."""
        return self.vecstore_retriever, self.parent_retriever

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        library_name: str,
        **kwargs,
    ) -> FleetContextRetrieverMachine:
        """Create FleetContextRetrieverMachine from df."""
        return cls(df, library_name=library_name, **kwargs)

    @classmethod
    def from_library(
        cls,
        library_name: str,
        download_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> FleetContextRetrieverMachine:
        """Create FleetContextRetrieverMachine from library_name."""
        download_kwargs = download_kwargs or {}
        try:
            library_df = download_embeddings(library_name, **download_kwargs)
        except TypeError:
            if download_kwargs:
                warnings.warn(
                    "`download_kwargs` not yet implemented in `context`; ignoring.",
                )
            library_df = download_embeddings(library_name)
        return cls(library_df, library_name=library_name, **kwargs)

    @classmethod
    def from_parquet(cls, filename: str, **kwargs) -> FleetContextRetrieverMachine:
        """Create FleetContextRetrieverMachine from parquet filename."""
        filename_pat = re.compile("libraries_(.*).parquet")

        search_result = filename_pat.search(filename)
        if search_result is None:
            raise ValueError(
                f"filename {filename} does not match pattern {filename_pat}",
            )
        library_name = search_result.group(1)
        return cls(pd.read_parquet(filename), library_name=library_name, **kwargs)

    @classmethod
    def retrievers_from_df(
        cls,
        df: pd.DataFrame,
        library_name: str,
        **kwargs,
    ) -> tuple[VectorStoreRetriever, MultiVectorRetriever]:
        """Create retrievers from df."""
        return cls.from_df(df, library_name=library_name, **kwargs).retrievers()

    @classmethod
    def retrievers_from_library(
        cls,
        library_name: str,
        download_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> tuple[VectorStoreRetriever, MultiVectorRetriever]:
        """Create retrievers from library_name."""
        return cls.from_library(library_name, download_kwargs, **kwargs).retrievers()

    @classmethod
    def retrievers_from_parquet(
        cls,
        filename: str,
        **kwargs,
    ) -> tuple[VectorStoreRetriever, MultiVectorRetriever]:
        """Create retrievers from parquet filename."""
        return cls.from_parquet(filename, **kwargs).retrievers()
