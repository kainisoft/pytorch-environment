"""
Database base configuration and declarative base.
"""
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

# Create a custom MetaData instance with naming convention
# This helps with consistent constraint naming for migrations
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

# Create the declarative base
Base = declarative_base(metadata=metadata)