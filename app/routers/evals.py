from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Depends
from typing import List, Dict
import logging

from core.schemas import (
    EvaluationQuery, EvaluationResult, ConsistencyCheck, MemoryQuery
)
from core.consistency import MemoryConsistencyChecker
from core.retrieval import HybridMemoryRetriever
from app.deps import db_manager, embedding_service

logger = logging.getLogger(__name__)

router = APIRouter()


def get_retriever(request: Request) -> HybridMemoryRetriever:
    """Get retriever from app state"""
    return request.app.state.retriever


def get_consistency_checker(request: Request) -> MemoryConsistencyChecker:
    """Get consistency checker from app state"""
    return request.app.state.consistency_checker


@router.post("/eval/precision-recall", response_model=EvaluationResult)
async def evaluate_precision_recall(
    eval_query: EvaluationQuery,
    retriever: HybridMemoryRetriever = Depends(get_retriever)
) -> EvaluationResult:
    """Evaluate precision and recall for a query"""
    try:
        # Perform the search
        memory_query = MemoryQuery(
            query_text=eval_query.query_text,
            limit=20  # Get more results for evaluation
        )

        search_response = await retriever.search_memories(memory_query)

        # Extract found memory IDs
        found_memory_ids = [result.memory.id for result in search_response.results]

        # Calculate precision and recall at different k values
        k_values = [1, 5, 10]
        precision_at_k = {}
        recall_at_k = {}

        expected_set = set(eval_query.expected_memory_ids)
        total_expected = len(expected_set)

        for k in k_values:
            # Get top-k results
            top_k_ids = found_memory_ids[:k]
            top_k_set = set(top_k_ids)

            # Calculate precision@k: relevant items in top-k / k
            relevant_in_top_k = len(top_k_set.intersection(expected_set))
            precision_at_k[k] = relevant_in_top_k / k if k > 0 else 0.0

            # Calculate recall@k: relevant items in top-k / total relevant
            recall_at_k[k] = relevant_in_top_k / total_expected if total_expected > 0 else 0.0

        # Calculate Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for expected_id in expected_set:
            if expected_id in found_memory_ids:
                rank = found_memory_ids.index(expected_id) + 1
                mrr += 1.0 / rank
                break  # Only consider the first relevant result

        # Find missing memories
        found_set = set(found_memory_ids)
        missing_memories = list(expected_set - found_set)

        return EvaluationResult(
            query_id=f"eval_{hash(eval_query.query_text)}",
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            found_memories=found_memory_ids,
            missing_memories=missing_memories
        )

    except Exception as e:
        logger.error(f"Precision/recall evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/eval/consistency", response_model=List[ConsistencyCheck])
async def evaluate_consistency(
    memory_ids: List[int] = None,
    consistency_checker: MemoryConsistencyChecker = Depends(get_consistency_checker)
) -> List[ConsistencyCheck]:
    """Evaluate consistency of memories"""
    try:
        if memory_ids:
            # Check consistency for specific memories
            # This is a simplified implementation
            # In practice, you'd implement specific consistency checks for given memories
            return []
        else:
            # Check all consistency
            return await consistency_checker.check_all_consistency()

    except Exception as e:
        logger.error(f"Consistency evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Consistency evaluation failed: {str(e)}")


@router.get("/eval/flagged-memories")
async def get_flagged_memories(
    consistency_checker: MemoryConsistencyChecker = Depends(get_consistency_checker)
):
    """Get memories flagged for review"""
    try:
        flagged_ids = await consistency_checker.get_flagged_memories()

        # Get details of flagged memories
        flagged_details = []
        if flagged_ids:
            async with db_manager.get_connection() as conn:
                query = """
                SELECT m.id, m.content, m.summary, m.chapter_number,
                       cl.change_reason, cl.created_at as flagged_at
                FROM memories m
                JOIN consistency_ledger cl ON m.id = cl.memory_id
                WHERE m.id = ANY($1) AND cl.flagged_for_review = true
                ORDER BY cl.created_at DESC
                """

                rows = await conn.fetch(query, flagged_ids)
                for row in rows:
                    flagged_details.append({
                        "memory_id": row['id'],
                        "content": row['content'][:200] + "..." if len(row['content']) > 200 else row['content'],
                        "summary": row['summary'],
                        "chapter_number": row['chapter_number'],
                        "reason": row['change_reason'],
                        "flagged_at": row['flagged_at']
                    })

        return {
            "total_flagged": len(flagged_ids),
            "flagged_memories": flagged_details
        }

    except Exception as e:
        logger.error(f"Failed to get flagged memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get flagged memories: {str(e)}")


@router.post("/eval/run-consistency-check")
async def run_consistency_check(
    background_tasks: BackgroundTasks,
    consistency_checker: MemoryConsistencyChecker = Depends(get_consistency_checker)
):
    """Run a comprehensive consistency check"""
    background_tasks.add_task(_run_consistency_check_task, consistency_checker)
    return {"message": "Consistency check started in background"}


async def _run_consistency_check_task(consistency_checker: MemoryConsistencyChecker):
    """Background task for consistency checking"""
    try:
        logger.info("Starting comprehensive consistency check")
        issues = await consistency_checker.check_all_consistency()

        # Flag memories with severe issues
        for issue in issues:
            if issue.consistency_score < 0.5:
                await consistency_checker.flag_memory_for_review(
                    issue.memory_id,
                    f"Consistency issue: {issue.explanation}"
                )

        logger.info(f"Consistency check completed. Found {len(issues)} issues")

    except Exception as e:
        logger.error(f"Consistency check task failed: {e}")


@router.get("/eval/system-metrics")
async def get_system_metrics():
    """Get evaluation metrics for the system"""
    try:
        async with db_manager.get_connection() as conn:
            # Memory distribution metrics
            memory_stats = await conn.fetch("""
                SELECT
                    memory_type,
                    COUNT(*) as count,
                    AVG(importance_score) as avg_importance,
                    AVG(access_count) as avg_access_count
                FROM memories
                GROUP BY memory_type
            """)

            # Character activity metrics
            character_stats = await conn.fetch("""
                SELECT
                    c.name,
                    COUNT(m.id) as memory_count,
                    AVG(m.importance_score) as avg_importance,
                    MAX(m.chapter_number) as last_chapter
                FROM characters c
                LEFT JOIN memories m ON c.id = m.character_id
                GROUP BY c.id, c.name
                ORDER BY memory_count DESC
            """)

            # Chapter progression metrics
            chapter_stats = await conn.fetch("""
                SELECT
                    chapter_number,
                    COUNT(*) as memory_count,
                    COUNT(DISTINCT character_id) as active_characters
                FROM memories
                WHERE chapter_number IS NOT NULL
                GROUP BY chapter_number
                ORDER BY chapter_number
            """)

            # Consistency metrics
            consistency_stats = await conn.fetch("""
                SELECT
                    change_type,
                    COUNT(*) as count,
                    SUM(CASE WHEN flagged_for_review THEN 1 ELSE 0 END) as flagged_count
                FROM consistency_ledger
                GROUP BY change_type
            """)

            return {
                "memory_distribution": [dict(row) for row in memory_stats],
                "character_activity": [dict(row) for row in character_stats],
                "chapter_progression": [dict(row) for row in chapter_stats],
                "consistency_metrics": [dict(row) for row in consistency_stats]
            }

    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


@router.post("/eval/benchmark")
async def run_benchmark_suite(
    background_tasks: BackgroundTasks,
    retriever: HybridMemoryRetriever = Depends(get_retriever)
):
    """Run a comprehensive benchmark of the system"""
    background_tasks.add_task(_run_benchmark_task, retriever)
    return {"message": "Benchmark suite started in background"}


async def _run_benchmark_task(retriever: HybridMemoryRetriever):
    """Background task for running benchmarks"""
    try:
        logger.info("Starting benchmark suite")

        # Define test queries for different scenarios
        test_queries = [
            {
                "query": "Byleth's relationship with Dimitri",
                "expected_tags": ["romantic", "professional"],
                "expected_characters": ["Byleth", "Dimitri"]
            },
            {
                "query": "Secret meetings and affairs",
                "expected_tags": ["secretive", "romantic"],
                "expected_characters": ["Byleth", "Dimitri", "Sylvain"]
            },
            {
                "query": "Annette's surprise plans",
                "expected_tags": ["romantic"],
                "expected_characters": ["Annette", "Sylvain"]
            },
            {
                "query": "Office interactions and work",
                "expected_tags": ["professional"],
                "expected_characters": ["Byleth", "Dimitri", "Sylvain"]
            }
        ]

        results = []
        for test in test_queries:
            query = MemoryQuery(
                query_text=test["query"],
                limit=10
            )

            response = await retriever.search_memories(query)

            # Analyze results
            found_tags = set()
            found_characters = set()

            for result in response.results:
                found_tags.update(result.memory.context_tags)
                # Would need character name mapping for proper evaluation

            result_analysis = {
                "query": test["query"],
                "total_found": len(response.results),
                "avg_similarity": sum(r.similarity_score for r in response.results) / len(response.results) if response.results else 0,
                "found_tags": list(found_tags),
                "expected_tags": test["expected_tags"],
                "tag_overlap": len(set(test["expected_tags"]).intersection(found_tags))
            }

            results.append(result_analysis)

        logger.info(f"Benchmark completed with {len(results)} test queries")

        # Store results (in practice, you might save to a file or database)
        return results

    except Exception as e:
        logger.error(f"Benchmark task failed: {e}")


@router.get("/eval/memory-relationships")
async def analyze_memory_relationships():
    """Analyze relationships between memories"""
    try:
        async with db_manager.get_connection() as conn:
            # Get relationship statistics
            relationship_stats = await conn.fetch("""
                SELECT
                    relationship_type,
                    COUNT(*) as count,
                    AVG(confidence_score) as avg_confidence
                FROM memory_relationships
                GROUP BY relationship_type
            """)

            # Get most connected memories
            connected_memories = await conn.fetch("""
                SELECT
                    m.id,
                    m.content,
                    m.summary,
                    COUNT(mr.source_memory_id) + COUNT(mr2.target_memory_id) as connection_count
                FROM memories m
                LEFT JOIN memory_relationships mr ON m.id = mr.source_memory_id
                LEFT JOIN memory_relationships mr2 ON m.id = mr2.target_memory_id
                GROUP BY m.id, m.content, m.summary
                HAVING COUNT(mr.source_memory_id) + COUNT(mr2.target_memory_id) > 0
                ORDER BY connection_count DESC
                LIMIT 10
            """)

            return {
                "relationship_statistics": [dict(row) for row in relationship_stats],
                "most_connected_memories": [dict(row) for row in connected_memories]
            }

    except Exception as e:
        logger.error(f"Failed to analyze memory relationships: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze relationships: {str(e)}")