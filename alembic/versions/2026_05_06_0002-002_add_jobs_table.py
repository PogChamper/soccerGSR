"""Add async jobs table for GSR pipeline

Revision ID: 002
Revises: 001
Create Date: 2026-05-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="queued"),
        sa.Column("stage", sa.String(length=20), nullable=True),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("input_filename", sa.String(length=255), nullable=True),
        sa.Column("input_path", sa.Text(), nullable=True),
        sa.Column("output_video_path", sa.Text(), nullable=True),
        sa.Column("gsr_json", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_id"), "jobs", ["id"], unique=False)
    op.create_index(op.f("ix_jobs_job_id"), "jobs", ["job_id"], unique=True)
    op.create_index(op.f("ix_jobs_created_at"), "jobs", ["created_at"], unique=False)
    op.create_index(op.f("ix_jobs_status"), "jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_jobs_status"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_created_at"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_job_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_id"), table_name="jobs")
    op.drop_table("jobs")
