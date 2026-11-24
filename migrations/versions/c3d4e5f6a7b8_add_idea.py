"""add idea ("Ý tưởng mỗi ngày" của Xưởng kịch bản)

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-11-24 23:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3d4e5f6a7b8'
down_revision = 'b2c3d4e5f6a7'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'idea',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('series_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False, server_default=''),
        sa.Column('angle', sa.Text(), nullable=False, server_default=''),
        sa.Column('key_points', sa.JSON(), nullable=False),
        sa.Column('sources', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=12), nullable=False, server_default='new'),
        sa.Column('episode_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['series_id'], ['series.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['episode_id'], ['episode.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('idea', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_idea_series_id'), ['series_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_idea_created_at'), ['created_at'], unique=False)


def downgrade():
    with op.batch_alter_table('idea', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_idea_created_at'))
        batch_op.drop_index(batch_op.f('ix_idea_series_id'))
    op.drop_table('idea')
