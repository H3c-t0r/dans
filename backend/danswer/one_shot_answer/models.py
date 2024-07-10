from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import root_validator

from danswer.chat.models import CitationInfo
from danswer.chat.models import DanswerContexts
from danswer.chat.models import DanswerQuotes
from danswer.chat.models import QADocsResponse
from danswer.configs.constants import MessageType
from danswer.db.models import StarterMessage
from danswer.search.enums import RecencyBiasSetting
from danswer.search.enums import SearchType
from danswer.search.models import ChunkContext
from danswer.search.models import RetrievalDetails


class QueryRephrase(BaseModel):
    rephrased_query: str


class ThreadMessage(BaseModel):
    message: str
    sender: str | None
    role: MessageType = MessageType.USER


class PromptConfig(BaseModel):
    name: str
    description: str = ""
    system_prompt: str
    task_prompt: str = ""
    include_citations: bool = True
    datetime_aware: bool = True


class DocumentSetConfig(BaseModel):
    id: int


class ToolConfig(BaseModel):
    id: int | None = None


class PersonaConfig(BaseModel):
    name: str
    description: str
    search_type: SearchType = SearchType.HYBRID
    num_chunks: float | None = None
    llm_relevance_filter: bool = False
    llm_filter_extraction: bool = False
    recency_bias: RecencyBiasSetting = RecencyBiasSetting.AUTO
    llm_model_provider_override: str | None = None
    llm_model_version_override: str | None = None
    starter_messages: list[StarterMessage] | None = None
    default_persona: bool = False
    is_visible: bool = True
    display_priority: int | None = None
    deleted: bool = False
    is_public: bool = True
    prompts: list[PromptConfig] = []
    document_sets: list[DocumentSetConfig] = []
    tools: list[ToolConfig] = []
    custom_tools_openapi: list[dict[str, Any]] | None = None


class DirectQARequest(ChunkContext):
    persona_config: PersonaConfig | None = None
    persona_id: int | None = None

    messages: list[ThreadMessage]
    prompt_id: int | None

    retrieval_options: RetrievalDetails = Field(default_factory=RetrievalDetails)
    # This is to forcibly skip (or run) the step, if None it uses the system defaults
    skip_rerank: bool | None = None
    skip_llm_chunk_filter: bool | None = None
    chain_of_thought: bool = False
    return_contexts: bool = False

    @root_validator
    def check_chain_of_thought_and_prompt_id(
        cls, values: dict[str, Any]
    ) -> dict[str, Any]:
        chain_of_thought = values.get("chain_of_thought")
        prompt_id = values.get("prompt_id")

        if chain_of_thought and prompt_id is not None:
            raise ValueError(
                "If chain_of_thought is True, prompt_id must be None"
                "The chain of thought prompt is only for question "
                "answering and does not accept customizing."
            )

        return values


class OneShotQAResponse(BaseModel):
    # This is built piece by piece, any of these can be None as the flow could break
    answer: str | None = None
    rephrase: str | None = None
    quotes: DanswerQuotes | None = None
    citations: list[CitationInfo] | None = None
    docs: QADocsResponse | None = None
    llm_chunks_indices: list[int] | None = None
    error_msg: str | None = None
    answer_valid: bool = True  # Reflexion result, default True if Reflexion not run
    chat_message_id: int | None = None
    contexts: DanswerContexts | None = None
