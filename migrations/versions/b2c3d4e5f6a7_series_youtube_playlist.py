"""add series.youtube_playlist; episode.task dài 40 (đủ cho "youtube:<lang>:<giờ đăng>")

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-11-24 23:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f6a7'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('series', schema=None) as batch_op:
        batch_op.add_column(sa.Column('youtube_playlist', sa.String(length=64), nullable=True))
    with op.batch_alter_table('episode', schema=None) as batch_op:
        batch_op.alter_column('task', existing_type=sa.String(length=20), type_=sa.String(length=40),
                              existing_nullable=True)


def downgrade():
    with op.batch_alter_table('episode', schema=None) as batch_op:
        batch_op.alter_column('task', existing_type=sa.String(length=40), type_=sa.String(length=20),
                              existing_nullable=True)
    with op.batch_alter_table('series', schema=None) as batch_op:
        batch_op.drop_column('youtube_playlist')
