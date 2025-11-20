import logging
from typing import List, Dict, Any, Callable
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from qdrant_client.http.exceptions import ResponseHandlingException
from config import qdrant, QDRANT_MAX_RETRIES

logger = logging.getLogger("qdrant_wrapper")

def _log_retry(retry_state):
    attempt = retry_state.attempt_number
    exception = retry_state.outcome.exception()
    logger.warning(f"Qdrant attempt {attempt} failed: {exception}")

@retry(
    reraise=True,
    stop=stop_after_attempt(QDRANT_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ResponseHandlingException),
    before_sleep=_log_retry
)
def safe_upsert(collection: str, points: List[Dict[str, Any]]):
    return qdrant.upsert(collection_name=collection, points=points, wait=True)

@retry(
    reraise=True,
    stop=stop_after_attempt(QDRANT_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(ResponseHandlingException),
    before_sleep=_log_retry
)
def safe_search(**kwargs):
    return qdrant.search(**kwargs)