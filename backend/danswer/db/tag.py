from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from danswer.configs.constants import DocumentSource
from danswer.db.models import Document
from danswer.db.models import Document__Tag
from danswer.db.models import Tag
from danswer.utils.logger import setup_logger

logger = setup_logger()


def create_or_add_document_tag(
    tag_key: str,
    tag_value: str,
    source: DocumentSource,
    document_id: str,
    db_session: Session,
) -> Tag:
    document = db_session.get(Document, document_id)
    if not document:
        raise ValueError("Invalid Document, cannot attach Tags")

    tag_stmt = select(Tag).where(
        Tag.tag_key == tag_key,
        Tag.tag_value == tag_value,
        Tag.source == source,
    )
    tag = db_session.execute(tag_stmt).scalar_one_or_none()

    if not tag:
        tag = Tag(tag_key=tag_key, tag_value=tag_value, source=source)
        db_session.add(tag)

    if tag not in document.tags:
        document.tags.append(tag)

    db_session.commit()
    return tag


def create_or_add_document_tag_list(
    tag_key: str,
    tag_values: list[str],
    source: DocumentSource,
    document_id: str,
    db_session: Session,
) -> list[Tag]:
    document = db_session.get(Document, document_id)
    if not document:
        raise ValueError("Invalid Document, cannot attach Tags")

    existing_tags_stmt = select(Tag).where(
        Tag.tag_key == tag_key, Tag.tag_value.in_(tag_values), Tag.source == source
    )
    existing_tags = list(db_session.execute(existing_tags_stmt).scalars().all())
    existing_tag_values = {tag.tag_value for tag in existing_tags}

    new_tags = []
    for tag_value in tag_values:
        if tag_value not in existing_tag_values:
            new_tag = Tag(tag_key=tag_key, tag_value=tag_value, source=source)
            db_session.add(new_tag)
            new_tags.append(new_tag)

    all_tags = existing_tags + new_tags

    for tag in all_tags:
        if tag not in document.tags:
            document.tags.append(tag)

    db_session.commit()
    return all_tags


def get_tags_by_value_prefix_for_source_types(
    tag_value_prefix: str | None,
    sources: list[DocumentSource] | None,
    db_session: Session,
) -> list[Tag]:
    query = select(Tag)

    if tag_value_prefix:
        query = query.where(Tag.tag_value.startswith(tag_value_prefix))

    if sources:
        query = query.where(Tag.source.in_(sources))

    result = db_session.execute(query)

    tags = result.scalars().all()
    return list(tags)


def delete_document_tags_for_documents(
    document_ids: list[str], db_session: Session
) -> None:
    """NOTE: does not commit transaction so that this can be used as part of a
    larger transaction block."""
    stmt = delete(Document__Tag).where(Document__Tag.document_id.in_(document_ids))
    db_session.execute(stmt)
