from collections.abc import Callable

from danswer.chat.chat_utils import combine_message_chain
from danswer.configs.chat_configs import DISABLE_LLM_QUERY_REPHRASE
from danswer.configs.model_configs import GEN_AI_HISTORY_CUTOFF
from danswer.db.models import ChatMessage
from danswer.llm.answering.models import PreviousMessage
from danswer.llm.exceptions import GenAIDisabledException
from danswer.llm.factory import get_default_llms
from danswer.llm.interfaces import LLM
from danswer.llm.utils import dict_based_prompt_to_langchain_prompt
from danswer.llm.utils import message_to_string
from danswer.prompts.chat_prompts import HISTORY_QUERY_REPHRASE
from danswer.prompts.miscellaneous_prompts import LANGUAGE_REPHRASE_PROMPT
from danswer.utils.logger import setup_logger
from danswer.utils.text_processing import count_punctuation
from danswer.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()


def llm_multilingual_query_expansion(query: str, language: str) -> str:
    def _get_rephrase_messages() -> list[dict[str, str]]:
        messages = [
            {
                "role": "user",
                "content": LANGUAGE_REPHRASE_PROMPT.format(
                    query=query, target_language=language
                ),
            },
        ]

        return messages

    try:
        _, fast_llm = get_default_llms(timeout=5)
    except GenAIDisabledException:
        logger.warning(
            "Unable to perform multilingual query expansion, Gen AI disabled"
        )
        return query

    messages = _get_rephrase_messages()
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
    model_output = message_to_string(fast_llm.invoke(filled_llm_prompt))
    logger.debug(model_output)

    return model_output


def multilingual_query_expansion(
    query: str,
    expansion_languages: str,
    use_threads: bool = True,
) -> list[str]:
    languages = expansion_languages.split(",")
    languages = [language.strip() for language in languages]
    if use_threads:
        functions_with_args: list[tuple[Callable, tuple]] = [
            (llm_multilingual_query_expansion, (query, language))
            for language in languages
        ]

        query_rephrases = run_functions_tuples_in_parallel(functions_with_args)
        return query_rephrases

    else:
        query_rephrases = [
            llm_multilingual_query_expansion(query, language) for language in languages
        ]
        return query_rephrases


def get_contextual_rephrase_messages(
    question: str,
    history_str: str,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "user",
            "content": HISTORY_QUERY_REPHRASE.format(
                question=question, chat_history=history_str
            ),
        },
    ]

    return messages


def history_based_query_rephrase(
    query: str,
    history: list[ChatMessage] | list[PreviousMessage],
    llm: LLM,
    size_heuristic: int = 200,
    punctuation_heuristic: int = 10,
    skip_first_rephrase: bool = False,
) -> str:
    # Globally disabled, just use the exact user query
    if DISABLE_LLM_QUERY_REPHRASE:
        return query

    # For some use cases, the first query should be untouched. Later queries must be rephrased
    # due to needing context but the first query has no context.
    if skip_first_rephrase and not history:
        return query

    # If it's a very large query, assume it's a copy paste which we may want to find exactly
    # or at least very closely, so don't rephrase it
    if len(query) >= size_heuristic:
        return query

    # If there is an unusually high number of punctuations, it's probably not natural language
    # so don't rephrase it
    if count_punctuation(query) >= punctuation_heuristic:
        return query

    history_str = combine_message_chain(
        messages=history, token_limit=GEN_AI_HISTORY_CUTOFF
    )

    prompt_msgs = get_contextual_rephrase_messages(
        question=query, history_str=history_str
    )

    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(prompt_msgs)
    rephrased_query = message_to_string(llm.invoke(filled_llm_prompt))

    logger.debug(f"Rephrased combined query: {rephrased_query}")

    return rephrased_query


def thread_based_query_rephrase(
    user_query: str,
    history_str: str,
    llm: LLM | None = None,
    size_heuristic: int = 200,
    punctuation_heuristic: int = 10,
) -> str:
    if not history_str:
        return user_query

    if len(user_query) >= size_heuristic:
        return user_query

    if count_punctuation(user_query) >= punctuation_heuristic:
        return user_query

    if llm is None:
        try:
            llm, _ = get_default_llms()
        except GenAIDisabledException:
            # If Generative AI is turned off, just return the original query
            return user_query

    prompt_msgs = get_contextual_rephrase_messages(
        question=user_query, history_str=history_str
    )

    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(prompt_msgs)
    rephrased_query = message_to_string(llm.invoke(filled_llm_prompt))

    logger.debug(f"Rephrased combined query: {rephrased_query}")

    return rephrased_query
