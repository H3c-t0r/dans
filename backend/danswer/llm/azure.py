from typing import Any

from langchain.chat_models.azure_openai import AzureChatOpenAI

from danswer.configs.model_configs import API_BASE_OPENAI
from danswer.configs.model_configs import API_VERSION_OPENAI
from danswer.configs.model_configs import AZURE_DEPLOYMENT_ID
from danswer.llm.llm import LangChainChatLLM
from danswer.llm.utils import should_be_verbose


class AzureGPT(LangChainChatLLM):
    def __init__(
        self,
        api_key: str,
        max_output_tokens: int,
        timeout: int,
        model_version: str,
        api_base: str = API_BASE_OPENAI,
        api_version: str = API_VERSION_OPENAI,
        deployment_name: str = AZURE_DEPLOYMENT_ID,
        *args: list[Any],
        **kwargs: dict[str, Any]
    ):
        self._llm = AzureChatOpenAI(
            model=model_version,
            openai_api_type="azure",
            openai_api_base=api_base,
            openai_api_version=api_version,
            deployment_name=deployment_name,
            openai_api_key=api_key,
            max_tokens=max_output_tokens,
            temperature=0,
            request_timeout=timeout,
            model_kwargs={
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            },
            verbose=should_be_verbose(),
        )

    @property
    def llm(self) -> AzureChatOpenAI:
        return self._llm
