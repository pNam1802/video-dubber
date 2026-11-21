"""add user.quota_gpu_seconds_per_day (per-user GPU override)

Revision ID: b9c0d1e2f3a4
Revises: a8b9c0d1e2f3
Create Date: 2026-11-21 07:00:00

"""
from alembic import op
import sqlalchemy as sa


revision = 'b9c0d1e2f3a4'
down_revision = 'a8b9c0d1e2f3'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('quota_gpu_seconds_per_day', sa.Integer(), nullable=True))


def downgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_column('quota_gpu_seconds_per_day')
