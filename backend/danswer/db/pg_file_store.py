from io import BytesIO
from typing import IO

from psycopg2.extensions import connection
from sqlalchemy.orm import Session

from danswer.db.models import PGFileStore
from danswer.utils.logger import setup_logger

logger = setup_logger()


def get_pg_conn_from_session(db_session: Session) -> connection:
    return db_session.connection().connection.connection  # type: ignore


def create_populate_lobj(
    content: IO,
    db_session: Session,
) -> int:
    """Note, this does not commit the changes to the DB
    This is because the commit should happen with the PGFileStore row creation
    That step finalizes both the Large Object and the table tracking it
    """
    pg_conn = get_pg_conn_from_session(db_session)
    large_object = pg_conn.lobject()

    large_object.write(content.read())
    large_object.close()

    return large_object.oid


def read_lobj(lobj_oid: int, db_session: Session, mode: str | None = None) -> IO:
    pg_conn = get_pg_conn_from_session(db_session)
    large_object = (
        pg_conn.lobject(lobj_oid, mode=mode) if mode else pg_conn.lobject(lobj_oid)
    )
    return BytesIO(large_object.read())


def delete_lobj_by_id(
    lobj_oid: int,
    db_session: Session,
) -> None:
    pg_conn = get_pg_conn_from_session(db_session)
    pg_conn.lobject(lobj_oid).unlink()


def upsert_pgfilestore(
    file_name: str,
    lobj_oid: int,
    db_session: Session,
) -> PGFileStore:
    pgfilestore = db_session.query(PGFileStore).filter_by(file_name=file_name).first()

    if pgfilestore:
        try:
            # This should not happen in normal execution
            delete_lobj_by_id(lobj_oid=pgfilestore.lobj_oid, db_session=db_session)
        except Exception:
            # If the delete fails as well, the large object doesn't exist anyway and even if it
            # fails to delete, it's not too terrible as most files sizes are insignificant
            logger.error(
                f"Failed to delete large object with oid {pgfilestore.lobj_oid}"
            )

        pgfilestore.lobj_oid = lobj_oid
    else:
        pgfilestore = PGFileStore(file_name=file_name, lobj_oid=lobj_oid)
        db_session.add(pgfilestore)

    db_session.commit()

    return pgfilestore


def get_pgfilestore_by_file_name(
    file_name: str,
    db_session: Session,
) -> PGFileStore:
    pgfilestore = db_session.query(PGFileStore).filter_by(file_name=file_name).first()

    if not pgfilestore:
        raise RuntimeError(f"File by name {file_name} does not exist or was deleted")

    return pgfilestore


def delete_pgfilestore_by_file_name(
    file_name: str,
    db_session: Session,
) -> None:
    db_session.query(PGFileStore).filter_by(file_name=file_name).delete()
