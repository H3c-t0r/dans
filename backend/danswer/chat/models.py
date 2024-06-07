from collections.abc import Iterator
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from danswer.configs.constants import DocumentSource
from danswer.configs.model_configs import GEN_AI_MAX_OUTPUT_TOKENS
from danswer.search.enums import QueryFlow
from danswer.search.enums import SearchType
from danswer.search.models import RetrievalDocs
from danswer.search.models import SearchResponse


class MessageChunk(BaseModel):
    content: str
    tokens: int | None = None


class LlmDoc(BaseModel):
    """This contains the minimal set information for the LLM portion including citations"""

    document_id: str
    content: str
    blurb: str
    semantic_identifier: str
    source_type: DocumentSource
    metadata: dict[str, str | list[str]]
    updated_at: datetime | None
    link: str | None
    source_links: dict[int, str] | None


# First chunk of info for streaming QA
class QADocsResponse(RetrievalDocs):
    rephrased_query: str | None = None
    predicted_flow: QueryFlow | None
    predicted_search: SearchType | None
    applied_source_filters: list[DocumentSource] | None
    applied_time_cutoff: datetime | None
    recency_bias_multiplier: float

    def dict(self, *args: list, **kwargs: dict[str, Any]) -> dict[str, Any]:  # type: ignore
        initial_dict = super().dict(*args, **kwargs)  # type: ignore
        initial_dict["applied_time_cutoff"] = (
            self.applied_time_cutoff.isoformat() if self.applied_time_cutoff else None
        )
        return initial_dict


# Second chunk of info for streaming QA
class LLMRelevanceFilterResponse(BaseModel):
    relevant_chunk_indices: list[int]


class DanswerAnswerPiece(BaseModel):
    answer_piece: str | None  # if None, specifies the end of an Answer
    token_count: int | None = None
    max_token: bool | None = None  # default to None if not set

    @classmethod
    def build_from_token_cnt(
        cls, answer_piece: str | None, token_count: int | None
    ) -> "DanswerAnswerPiece":
        if token_count is None:
            token_count = 1
        return DanswerAnswerPiece(
            answer_piece=answer_piece,
            token_count=token_count,
            max_token=(
                token_count == GEN_AI_MAX_OUTPUT_TOKENS
                if token_count is not None
                else None
            ),
        )


# An intermediate representation of citations, later translated into
# a mapping of the citation [n] number to SearchDoc
class CitationInfo(BaseModel):
    citation_num: int
    document_id: str


class StreamingError(BaseModel):
    error: str


class DanswerQuote(BaseModel):
    # This is during inference so everything is a string by this point
    quote: str
    document_id: str
    link: str | None
    source_type: str
    semantic_identifier: str
    blurb: str


class DanswerQuotes(BaseModel):
    quotes: list[DanswerQuote]


class DanswerContext(BaseModel):
    content: str
    document_id: str
    semantic_identifier: str
    blurb: str


class DanswerContexts(BaseModel):
    contexts: list[DanswerContext]


class DanswerAnswer(BaseModel):
    answer: str | None


class QAResponse(SearchResponse, DanswerAnswer):
    quotes: list[DanswerQuote] | None
    contexts: list[DanswerContexts] | None
    predicted_flow: QueryFlow
    predicted_search: SearchType
    eval_res_valid: bool | None = None
    llm_chunks_indices: list[int] | None = None
    error_msg: str | None = None


class ImageGenerationDisplay(BaseModel):
    file_ids: list[str]


AnswerQuestionPossibleReturn = (
    DanswerAnswerPiece
    | DanswerQuotes
    | CitationInfo
    | DanswerContexts
    | ImageGenerationDisplay
    | StreamingError
)


AnswerQuestionStreamReturn = Iterator[AnswerQuestionPossibleReturn]


class LLMMetricsContainer(BaseModel):
    prompt_tokens: int
    response_tokens: int
