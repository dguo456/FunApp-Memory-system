import os
from functools import lru_cache
from typing import Generator, Optional, List, Tuple
import asyncpg
from sentence_transformers import SentenceTransformer
import numpy as np
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def init_pool(self):
        """Initialize database connection pool"""
        database_url = get_settings().database_url
        try:
            self.pool = await asyncpg.create_pool(
                database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("Database pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise

    async def close_pool(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            await self.init_pool()

        async with self.pool.acquire() as connection:
            yield connection


class EmbeddingService:
    def __init__(self):
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of the embedding model"""
        if self._model is None:
            try:
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._model

    def encode_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0


class Settings:
    def __init__(self):
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            f"postgresql://{os.getenv('USER', 'postgres')}@localhost:5432/sekai_memory"
        )
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "384"))
        self.max_memory_content_length: int = int(os.getenv("MAX_MEMORY_CONTENT_LENGTH", "2000"))
        self.default_similarity_threshold: float = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.7"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # LLM settings for consistency checking (optional)
        self.llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.llm_endpoint: Optional[str] = os.getenv("LLM_ENDPOINT")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global instances
db_manager = DatabaseManager()
embedding_service = EmbeddingService()


async def get_db_connection():
    """Dependency for getting database connection"""
    async with db_manager.get_connection() as connection:
        yield connection


def get_embedding_service() -> EmbeddingService:
    """Dependency for getting embedding service"""
    return embedding_service


async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up Sekai Memory System...")
    await db_manager.init_pool()

    # Warm up embedding service
    embedding_service.encode_text("warmup")
    logger.info("Services initialized successfully")


async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Sekai Memory System...")
    await db_manager.close_pool()
    logger.info("Cleanup completed")


# Setup logging
def setup_logging():
    logging.basicConfig(
        level=getattr(logging, get_settings().log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('sekai_memory.log')
        ]
    )


# Database utilities
async def execute_query(query: str, *args, fetch: bool = False):
    """Execute a database query"""
    async with db_manager.get_connection() as conn:
        if fetch:
            return await conn.fetch(query, *args)
        else:
            return await conn.execute(query, *args)


async def execute_transaction(queries: List[Tuple]):
    """Execute multiple queries in a transaction"""
    async with db_manager.get_connection() as conn:
        async with conn.transaction():
            results = []
            for query, args in queries:
                result = await conn.execute(query, *args)
                results.append(result)
            return results


# Health check utilities
async def check_database_health() -> bool:
    """Check if database is healthy"""
    try:
        async with db_manager.get_connection() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def check_embedding_service_health() -> bool:
    """Check if embedding service is healthy"""
    try:
        embedding_service.encode_text("health check")
        return True
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
        return False