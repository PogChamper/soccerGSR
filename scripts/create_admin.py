"""Create an admin user (or promote an existing one).

DELETE /history requires an admin JWT; registration via the API never grants
admin, so this is the only supported way to get one.

Usage:
    PYTHONPATH=. python scripts/create_admin.py <username> [password]

If the user exists, it is promoted to admin (password updated when given).
"""
from __future__ import annotations

import asyncio
import getpass
import sys


async def main(username: str, password: str | None) -> None:
    from sqlalchemy import select

    from app.api.auth import get_password_hash
    from app.models.database import User, async_session, init_db

    await init_db()
    async with async_session() as db:
        res = await db.execute(select(User).where(User.username == username))
        user = res.scalar_one_or_none()
        if user is None:
            if not password:
                raise SystemExit("password is required for a new user")
            user = User(
                username=username,
                password_hash=get_password_hash(password),
                is_admin=True,
            )
            db.add(user)
            action = "created"
        else:
            user.is_admin = True
            if password:
                user.password_hash = get_password_hash(password)
            action = "promoted"
        await db.commit()
    print(f"admin '{username}' {action}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    pwd = sys.argv[2] if len(sys.argv) > 2 else getpass.getpass("password (empty to keep): ") or None
    asyncio.run(main(sys.argv[1], pwd))
