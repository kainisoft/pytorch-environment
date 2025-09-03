"""
Database connection and session management.
"""
import logging
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# SQLite-specific configuration for foreign key support
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite."""
    if "sqlite" in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Synchronous database engine and session
engine = create_engine(
    settings.database_url,
    echo=settings.debug,  # Log SQL queries in debug mode
    pool_pre_ping=True,   # Verify connections before use
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# Async database engine and session (for future async operations)
async_engine = None
AsyncSessionLocal = None

if settings.database_url.startswith("postgresql"):
    # Only create async engine for PostgreSQL
    async_database_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
    async_engine = create_async_engine(
        async_database_url,
        echo=settings.debug,
        pool_pre_ping=True,
    )
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    
    Yields:
        Session: SQLAlchemy database session
        
    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db session here
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async dependency function to get database session.
    
    Yields:
        AsyncSession: SQLAlchemy async database session
        
    Usage:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_async_db)):
            # Use async db session here
    """
    if AsyncSessionLocal is None:
        raise RuntimeError("Async database not configured. Use PostgreSQL for async operations.")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Async database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


def create_tables():
    """
    Create all database tables.
    
    This function should be called during application startup
    to ensure all tables exist.
    """
    from app.database.base import Base
    
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_tables():
    """
    Drop all database tables.
    
    WARNING: This will delete all data. Use only for testing or reset.
    """
    from app.database.base import Base
    
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped")


async def create_tables_async():
    """
    Create all database tables asynchronously.
    
    This function should be used when working with async database operations.
    """
    if async_engine is None:
        raise RuntimeError("Async database not configured")
    
    from app.database.base import Base
    
    logger.info("Creating database tables asynchronously...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")


def check_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        bool: True if connection is successful, False otherwise
    """
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        logger.info("Database connection check successful")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False