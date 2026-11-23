"""add episode storyboard_vi/en + previews (duyệt phân cảnh Video giải thích)

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-11-24 00:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd2e3f4a5b6c7'
down_revision = 'c1d2e3f4a5b6'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('episode', schema=None) as batch_op:
        batch_op.add_column(sa.Column('storyboard_vi', sa.JSON(), nullable=False, server_default='[]'))
        batch_op.add_column(sa.Column('storyboard_en', sa.JSON(), nullable=False, server_default='[]'))
        batch_op.add_column(sa.Column('previews', sa.JSON(), nullable=False, server_default='{}'))


def downgrade():
    with op.batch_alter_table('episode', schema=None) as batch_op:
        batch_op.drop_column('previews')
        batch_op.drop_column('storyboard_en')
        batch_op.drop_column('storyboard_vi')
