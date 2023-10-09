from sqlalchemy.orm import Session

from danswer.document_index.factory import get_default_document_index
from danswer.document_index.interfaces import DocumentIndex
from danswer.document_index.interfaces import UpdateRequest
from danswer.db.document import prepare_to_modify_documents
from danswer.db.engine import get_sqlalchemy_engine
from danswer.utils.batching import batch_generator
from danswer.utils.logger import setup_logger
from ee.danswer.access.access import get_access_for_documents
from ee.danswer.db.user_group import delete_user_group
from ee.danswer.db.user_group import fetch_documents_for_user_group
from ee.danswer.db.user_group import fetch_user_group
from ee.danswer.db.user_group import mark_user_group_as_synced

logger = setup_logger()

_SYNC_BATCH_SIZE = 1000


def _sync_user_group_batch(
    document_ids: list[str], document_index: DocumentIndex
) -> None:
    logger.debug(f"Syncing document sets for: {document_ids}")
    # begin a transaction, release lock at the end
    with Session(get_sqlalchemy_engine()) as db_session:
        # acquires a lock on the documents so that no other process can modify them
        prepare_to_modify_documents(db_session=db_session, document_ids=document_ids)

        # get current state of document sets for these documents
        document_id_to_access = get_access_for_documents(
            document_ids=document_ids, db_session=db_session
        )

        # update Vespa
        document_index.update(
            update_requests=[
                UpdateRequest(
                    document_ids=[document_id],
                    access=document_id_to_access[document_id],
                )
                for document_id in document_ids
            ]
        )


def sync_user_groups(user_group_id: int) -> None:
    """Sync the status of Postgres for the specified user group"""
    document_index = get_default_document_index()
    with Session(get_sqlalchemy_engine()) as db_session:
        user_group = fetch_user_group(
            db_session=db_session, user_group_id=user_group_id
        )
        if user_group is None:
            raise ValueError(f"User group '{user_group_id}' does not exist")

        documents_to_update = fetch_documents_for_user_group(
            db_session=db_session,
            user_group_id=user_group_id,
        )
        for document_batch in batch_generator(documents_to_update, _SYNC_BATCH_SIZE):
            _sync_user_group_batch(
                document_ids=[document.id for document in document_batch],
                document_index=document_index,
            )

        if user_group.is_up_for_deletion:
            delete_user_group(db_session=db_session, user_group=user_group)
        else:
            mark_user_group_as_synced(db_session=db_session, user_group=user_group)
