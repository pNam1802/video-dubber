"""add episode.publish (phụ đề hai thứ tiếng + gói đăng YouTube)

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-11-24 01:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e3f4a5b6c7d8'
down_revision = 'd2e3f4a5b6c7'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('episode', schema=None) as batch_op:
        batch_op.add_column(sa.Column('publish', sa.JSON(), nullable=False, server_default='{}'))


def downgrade():
    with op.batch_alter_table('episode', schema=None) as batch_op:
        batch_op.drop_column('publish')
