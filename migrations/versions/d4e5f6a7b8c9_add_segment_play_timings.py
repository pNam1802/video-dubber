"""add transcript_segment play_start_sec/play_end_sec (moc trong video da long tieng)

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-10-27 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd4e5f6a7b8c9'
down_revision = 'c3d4e5f6a7b8'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('transcript_segment', schema=None) as batch_op:
        batch_op.add_column(sa.Column('play_start_sec', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('play_end_sec', sa.Float(), nullable=True))


def downgrade():
    with op.batch_alter_table('transcript_segment', schema=None) as batch_op:
        batch_op.drop_column('play_end_sec')
        batch_op.drop_column('play_start_sec')
