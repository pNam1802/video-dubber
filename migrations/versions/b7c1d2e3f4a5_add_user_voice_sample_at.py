"""add user voice_sample_at (giong mau de VieNeu clone)

Revision ID: b7c1d2e3f4a5
Revises: 84ce1df9e6ce
Create Date: 2026-10-27 01:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b7c1d2e3f4a5'
down_revision = '84ce1df9e6ce'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('voice_sample_at', sa.DateTime(timezone=True), nullable=True))


def downgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_column('voice_sample_at')
