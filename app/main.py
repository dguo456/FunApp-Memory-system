import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import our modules
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(current_dir)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.deps import (
    startup_event, shutdown_event, setup_logging,
    db_manager, embedding_service,
    check_database_health, check_embedding_service_health
)
from app.routers import memories, evals
from core.schemas import SystemHealth, MemoryStats, MemoryBasic
from core.retrieval import HybridMemoryRetriever
from core.ingest import MemoryIngestionService
from core.consistency import MemoryConsistencyChecker

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    await startup_event()

    # Initialize services
    app.state.retriever = HybridMemoryRetriever(db_manager, embedding_service)
    app.state.ingestion_service = MemoryIngestionService(db_manager)
    app.state.consistency_checker = MemoryConsistencyChecker(db_manager, embedding_service)

    logger.info("Sekai Memory System started successfully")

    yield

    # Shutdown
    await shutdown_event()
    logger.info("Sekai Memory System shut down")


# Create FastAPI app
app = FastAPI(
    title="Sekai Memory System",
    description="Multi-character memory system for narrative applications",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(memories.router, prefix="/api/v1", tags=["memories"])
app.include_router(evals.router, prefix="/api/v1", tags=["evaluations"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sekai Memory System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=SystemHealth)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database health
        db_healthy = await check_database_health()
        db_status = "healthy" if db_healthy else "unhealthy"

        # Check embedding service health
        embedding_healthy = check_embedding_service_health()
        embedding_status = "healthy" if embedding_healthy else "unhealthy"

        # Get memory count
        total_memories = 0
        flagged_count = 0

        if db_healthy:
            async with db_manager.get_connection() as conn:
                total_memories = await conn.fetchval("SELECT COUNT(*) FROM memories")
                flagged_count = await conn.fetchval(
                    "SELECT COUNT(DISTINCT memory_id) FROM consistency_ledger WHERE flagged_for_review = true"
                )

        # Calculate overall consistency score (simplified)
        consistency_score = 1.0
        if total_memories > 0:
            consistency_score = max(0.0, 1.0 - (flagged_count / total_memories))

        return SystemHealth(
            database_status=db_status,
            embedding_service_status=embedding_status,
            total_memories=total_memories,
            flagged_for_review=flagged_count,
            last_update=await _get_last_update_time(),
            consistency_score=consistency_score
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/stats", response_model=MemoryStats)
async def get_system_stats():
    """Get system statistics"""
    try:
        async with db_manager.get_connection() as conn:
            # Total memories
            total_memories = await conn.fetchval("SELECT COUNT(*) FROM memories")

            # Memories by type
            type_stats = await conn.fetch(
                "SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type"
            )
            memories_by_type = {row['memory_type']: row['count'] for row in type_stats}

            # Memories by character
            char_stats = await conn.fetch("""
                SELECT c.name, COUNT(m.id) as count
                FROM characters c
                LEFT JOIN memories m ON c.id = m.character_id
                GROUP BY c.name
            """)
            memories_by_character = {row['name']: row['count'] for row in char_stats}

            # Average importance score
            avg_importance = await conn.fetchval(
                "SELECT AVG(importance_score) FROM memories"
            ) or 0.0

            # Most accessed memories
            most_accessed_rows = await conn.fetch("""
                SELECT id, content, summary, access_count
                FROM memories
                ORDER BY access_count DESC
                LIMIT 5
            """)

            most_accessed = []
            for row in most_accessed_rows:
                memory_basic = MemoryBasic(
                    id=row['id'],
                    content=row['content'],
                    summary=row['summary'],
                    access_count=row['access_count']
                )
                most_accessed.append(memory_basic)

            # Recent memories
            recent_rows = await conn.fetch("""
                SELECT id, content, summary, created_at
                FROM memories
                ORDER BY created_at DESC
                LIMIT 5
            """)

            recent_memories = []
            for row in recent_rows:
                memory_basic = MemoryBasic(
                    id=row['id'],
                    content=row['content'],
                    summary=row['summary'],
                    created_at=row['created_at']
                )
                recent_memories.append(memory_basic)

        return MemoryStats(
            total_memories=total_memories,
            memories_by_type=memories_by_type,
            memories_by_character=memories_by_character,
            avg_importance_score=float(avg_importance),
            most_accessed_memories=most_accessed,
            recent_memories=recent_memories
        )

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system statistics")


async def _get_last_update_time():
    """Get the timestamp of the last memory update"""
    try:
        async with db_manager.get_connection() as conn:
            last_update = await conn.fetchval(
                "SELECT MAX(updated_at) FROM memories"
            )
            return last_update

    except Exception:
        from datetime import datetime
        return datetime.now()


@app.post("/admin/consistency-check")
async def run_consistency_check(background_tasks: BackgroundTasks):
    """Run consistency check in the background"""
    background_tasks.add_task(_run_consistency_check_task)
    return {"message": "Consistency check started in background"}


async def _run_consistency_check_task():
    """Background task to run consistency check"""
    try:
        logger.info("Starting consistency check task")
        consistency_checker = MemoryConsistencyChecker(db_manager, embedding_service)
        issues = await consistency_checker.check_all_consistency()

        # Flag problematic memories
        for issue in issues:
            if issue.consistency_score < 0.5:  # Threshold for flagging
                await consistency_checker.flag_memory_for_review(
                    issue.memory_id,
                    issue.explanation
                )

        logger.info(f"Consistency check completed. Found {len(issues)} issues")

    except Exception as e:
        logger.error(f"Consistency check task failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )