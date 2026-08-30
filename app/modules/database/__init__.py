"""SQL database access.

    engine   SQLAlchemy engine + Schedule queries.
    seeder   Ports data/db_Tech.sql (T-SQL) to the SQLite file the app reads.

The original SQL Server script is kept at ``data/db_Tech.sql`` as the reference
for the schema and the seeding rules. The runtime database is SQLite so the same
artifact works locally and on Streamlit Community Cloud, which cannot reach a
SQL Server instance. See ENGINEERING.md section 6.2.
"""

from app.modules.database.engine import get_engine

__all__ = ["get_engine"]
