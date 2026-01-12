"""Initial migration - create users and request_history tables

Revision ID: 001
Revises: 
Create Date: 2024-12-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('is_admin', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    
    # Create request_history table
    op.create_table(
        'request_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('request_id', sa.String(length=36), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('input_filename', sa.String(length=255), nullable=True),
        sa.Column('input_size_mb', sa.Float(), nullable=True),
        sa.Column('input_width', sa.Integer(), nullable=True),
        sa.Column('input_height', sa.Integer(), nullable=True),
        sa.Column('input_duration', sa.Float(), nullable=True),
        sa.Column('input_fps', sa.Float(), nullable=True),
        sa.Column('input_frames', sa.Integer(), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.Column('frames_processed', sa.Integer(), nullable=True),
        sa.Column('total_detections', sa.Integer(), nullable=True),
        sa.Column('players_count', sa.Integer(), nullable=True),
        sa.Column('goalkeepers_count', sa.Integer(), nullable=True),
        sa.Column('referees_count', sa.Integer(), nullable=True),
        sa.Column('balls_count', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_request_history_id'), 'request_history', ['id'], unique=False)
    op.create_index(op.f('ix_request_history_request_id'), 'request_history', ['request_id'], unique=True)
    op.create_index(op.f('ix_request_history_timestamp'), 'request_history', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_request_history_timestamp'), table_name='request_history')
    op.drop_index(op.f('ix_request_history_request_id'), table_name='request_history')
    op.drop_index(op.f('ix_request_history_id'), table_name='request_history')
    op.drop_table('request_history')
    
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')

