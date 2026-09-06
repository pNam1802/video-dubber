"""multi-language: rename transcript columns, add job source/target language

Revision ID: 8f0d5ed871f0
Revises: ac36db2cb97b
Create Date: 2026-09-07 02:53:24.842577

Chuan bi cho dich sang Nhat/Trung/Han ngoai tieng Viet: doi ten cot
text_en/text_vi thanh text_source/text_target (trung lap, khong con cung
tieng Anh/tieng Viet), va them Job.source_language/target_language de biet
dung ngon ngu that su cua tung job.

QUAN TRONG: autogenerate coi doi ten la "them cot moi + xoa cot cu" — se MAT
HET du lieu transcript da luu. Da sua tay thanh alter_column(new_column_name)
de GIU NGUYEN du lieu, chi doi ten cot.
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8f0d5ed871f0'
down_revision = 'ac36db2cb97b'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('job', schema=None) as batch_op:
        batch_op.add_column(sa.Column('source_language', sa.String(length=8), nullable=True))
        batch_op.add_column(sa.Column('target_language', sa.String(length=8), server_default='vi', nullable=False))

    with op.batch_alter_table('transcript_segment', schema=None) as batch_op:
        batch_op.alter_column('text_en', new_column_name='text_source')
        batch_op.alter_column('text_vi', new_column_name='text_target')


def downgrade():
    with op.batch_alter_table('transcript_segment', schema=None) as batch_op:
        batch_op.alter_column('text_source', new_column_name='text_en')
        batch_op.alter_column('text_target', new_column_name='text_vi')

    with op.batch_alter_table('job', schema=None) as batch_op:
        batch_op.drop_column('target_language')
        batch_op.drop_column('source_language')
