"""add series + episode (Xưởng kịch bản: agent lên ý tưởng series YouTube)

Revision ID: c1d2e3f4a5b6
Revises: b9c0d1e2f3a4
Create Date: 2026-11-23 19:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c1d2e3f4a5b6'
down_revision = 'b9c0d1e2f3a4'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'series',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), server_default='', nullable=False),
        sa.Column('topic', sa.Text(), server_default='', nullable=False),
        sa.Column('audience', sa.Text(), server_default='', nullable=False),
        sa.Column('description', sa.Text(), server_default='', nullable=False),
        sa.Column('style_guide', sa.Text(), server_default='', nullable=False),
        sa.Column('glossary', sa.JSON(), nullable=False),
        sa.Column('task', sa.String(length=20), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_series_user_id', 'series', ['user_id'])
    op.create_table(
        'episode',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('series_id', sa.Integer(), nullable=False),
        sa.Column('number', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), server_default='', nullable=False),
        sa.Column('goal', sa.Text(), server_default='', nullable=False),
        sa.Column('key_points', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='planned', nullable=False),
        sa.Column('notes', sa.Text(), server_default='', nullable=False),
        sa.Column('sources', sa.JSON(), nullable=False),
        sa.Column('outline', sa.JSON(), nullable=False),
        sa.Column('script_vi', sa.Text(), server_default='', nullable=False),
        sa.Column('script_en', sa.Text(), server_default='', nullable=False),
        sa.Column('quiz', sa.JSON(), nullable=False),
        sa.Column('review', sa.JSON(), nullable=False),
        sa.Column('summary', sa.Text(), server_default='', nullable=False),
        sa.Column('job_vi_id', sa.Integer(), nullable=True),
        sa.Column('job_en_id', sa.Integer(), nullable=True),
        sa.Column('task', sa.String(length=20), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['series_id'], ['series.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['job_vi_id'], ['job.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['job_en_id'], ['job.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('series_id', 'number', name='uq_episode_series_number'),
    )
    op.create_index('ix_episode_series_id', 'episode', ['series_id'])


def downgrade():
    op.drop_index('ix_episode_series_id', table_name='episode')
    op.drop_table('episode')
    op.drop_index('ix_series_user_id', table_name='series')
    op.drop_table('series')
