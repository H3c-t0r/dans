import time
from http import HTTPStatus

from danswer.configs.app_configs import KEYWORD_MAX_HITS
from danswer.configs.constants import CONTENT
from danswer.configs.constants import SOURCE_LINKS
from danswer.direct_qa import get_default_backend_qa_model
from danswer.direct_qa.semantic_search import semantic_search
from danswer.server.models import KeywordResponse
from danswer.server.models import QAQuestion
from danswer.server.models import QAResponse
from danswer.server.models import ServerStatus
from danswer.utils.clients import TSClient
from danswer.utils.logging import setup_logger
from fastapi import APIRouter


logger = setup_logger()

router = APIRouter()


@router.get("/", response_model=ServerStatus)
@router.get("/status", response_model=ServerStatus)
def read_server_status():
    return {"status": HTTPStatus.OK.value}


@router.post("/direct-qa", response_model=QAResponse)
def direct_qa(question: QAQuestion):
    qa_model = get_default_backend_qa_model()
    query = question.query
    collection = question.collection

    logger.info(f"Received semantic query: {query}")
    start_time = time.time()

    ranked_chunks = semantic_search(collection, query)

    answer, quotes = qa_model.answer_question(query, ranked_chunks)

    logger.info(f"Total QA took {time.time() - start_time} seconds")

    return QAResponse(answer=answer, quotes=quotes)


@router.post("/keyword-search", response_model=KeywordResponse)
def keyword_search(question: QAQuestion):
    ts_client = TSClient.get_instance()
    query = question.query
    collection = question.collection

    logger.info(f"Received keyword query: {query}")
    start_time = time.time()

    search_results = ts_client.collections[collection].documents.search(
        {
            "q": query,
            "query_by": CONTENT,
            "per_page": KEYWORD_MAX_HITS,
            "limit_hits": KEYWORD_MAX_HITS,
        }
    )

    hits = search_results["hits"]
    sources = [hit["document"][SOURCE_LINKS][0] for hit in hits]

    total_time = time.time() - start_time
    logger.info(f"Total Keyword Search took {total_time} seconds")

    return KeywordResponse(results=sources)
