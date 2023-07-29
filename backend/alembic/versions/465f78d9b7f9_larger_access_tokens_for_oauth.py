"""Larger Access Tokens for OAUTH

Revision ID: 465f78d9b7f9
Revises: 3c5e35aa9af0
Create Date: 2023-07-18 17:33:40.365034

"""
import sqlalchemy as sa

from alembic import op


# revision identifiers, used by Alembic.
revision = "465f78d9b7f9"
down_revision = "3c5e35aa9af0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("oauth_account", "access_token", type_=sa.Text())


def downgrade() -> None:
    op.alter_column("oauth_account", "access_token", type_=sa.String(length=1024))
