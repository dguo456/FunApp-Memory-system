import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import our modules
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(current_dir)

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

from core.schemas import EvaluationQuery, EvaluationResult, MemoryQuery
from core.retrieval import HybridMemoryRetriever
from core.consistency import MemoryConsistencyChecker
from app.deps import DatabaseManager, EmbeddingService

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Comprehensive evaluation system for Sekai Memory System"""

    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.retriever = HybridMemoryRetriever(db_manager, embedding_service)
        self.consistency_checker = MemoryConsistencyChecker(db_manager, embedding_service)

    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        logger.info("Starting full evaluation suite")

        results = {
            "timestamp": datetime.now().isoformat(),
            "precision_recall": await self._evaluate_precision_recall(),
            "consistency": await self._evaluate_consistency(),
            "retrieval_quality": await self._evaluate_retrieval_quality(),
            "system_metrics": await self._get_system_metrics()
        }

        logger.info("Full evaluation suite completed")
        return results

    async def _evaluate_precision_recall(self) -> Dict[str, Any]:
        """Evaluate precision and recall using predefined queries"""
        test_queries = await self._load_test_queries()

        all_results = []
        total_precision = {1: 0, 5: 0, 10: 0}
        total_recall = {1: 0, 5: 0, 10: 0}
        total_mrr = 0

        for test_query in test_queries:
            try:
                # Convert to evaluation query
                eval_query = EvaluationQuery(
                    query_text=test_query["query"],
                    expected_memory_ids=test_query["expected_ids"],
                    query_type="precision_recall"
                )

                # Run evaluation
                result = await self._evaluate_single_query(eval_query)
                all_results.append({
                    "query": test_query["query"],
                    "result": result
                })

                # Accumulate metrics
                for k in [1, 5, 10]:
                    total_precision[k] += result.precision_at_k.get(k, 0)
                    total_recall[k] += result.recall_at_k.get(k, 0)
                total_mrr += result.mrr

            except Exception as e:
                logger.error(f"Failed to evaluate query '{test_query['query']}': {e}")

        # Calculate averages
        num_queries = len(test_queries)
        avg_precision = {k: total_precision[k] / num_queries for k in [1, 5, 10]} if num_queries > 0 else {}
        avg_recall = {k: total_recall[k] / num_queries for k in [1, 5, 10]} if num_queries > 0 else {}
        avg_mrr = total_mrr / num_queries if num_queries > 0 else 0

        return {
            "average_precision_at_k": avg_precision,
            "average_recall_at_k": avg_recall,
            "average_mrr": avg_mrr,
            "total_queries": num_queries,
            "individual_results": all_results
        }

    async def _evaluate_single_query(self, eval_query: EvaluationQuery) -> EvaluationResult:
        """Evaluate a single query for precision/recall"""
        # Perform search
        memory_query = MemoryQuery(
            query_text=eval_query.query_text,
            limit=20
        )

        search_response = await self.retriever.search_memories(memory_query)
        found_memory_ids = [result.memory.id for result in search_response.results]

        # Calculate metrics
        expected_set = set(eval_query.expected_memory_ids)
        k_values = [1, 5, 10]
        precision_at_k = {}
        recall_at_k = {}

        for k in k_values:
            top_k_ids = found_memory_ids[:k]
            top_k_set = set(top_k_ids)
            relevant_in_top_k = len(top_k_set.intersection(expected_set))

            precision_at_k[k] = relevant_in_top_k / k if k > 0 else 0.0
            recall_at_k[k] = relevant_in_top_k / len(expected_set) if len(expected_set) > 0 else 0.0

        # Calculate MRR
        mrr = 0.0
        for expected_id in expected_set:
            if expected_id in found_memory_ids:
                rank = found_memory_ids.index(expected_id) + 1
                mrr = 1.0 / rank
                break

        missing_memories = list(expected_set - set(found_memory_ids))

        return EvaluationResult(
            query_id=f"eval_{hash(eval_query.query_text)}",
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            found_memories=found_memory_ids,
            missing_memories=missing_memories
        )

    async def _evaluate_consistency(self) -> Dict[str, Any]:
        """Evaluate system consistency"""
        try:
            consistency_issues = await self.consistency_checker.check_all_consistency()

            # Categorize issues by severity
            severe_issues = [issue for issue in consistency_issues if issue.consistency_score < 0.3]
            moderate_issues = [issue for issue in consistency_issues if 0.3 <= issue.consistency_score < 0.7]
            minor_issues = [issue for issue in consistency_issues if issue.consistency_score >= 0.7]

            # Get flagged memories
            flagged_memories = await self.consistency_checker.get_flagged_memories()

            # Calculate overall consistency score
            total_memories = await self._get_total_memory_count()
            consistency_score = 1.0 - (len(consistency_issues) / total_memories) if total_memories > 0 else 1.0

            return {
                "overall_consistency_score": consistency_score,
                "total_issues": len(consistency_issues),
                "severe_issues": len(severe_issues),
                "moderate_issues": len(moderate_issues),
                "minor_issues": len(minor_issues),
                "flagged_memories": len(flagged_memories),
                "issue_details": [
                    {
                        "memory_id": issue.memory_id,
                        "consistency_score": issue.consistency_score,
                        "explanation": issue.explanation
                    }
                    for issue in consistency_issues[:10]  # Top 10 issues
                ]
            }

        except Exception as e:
            logger.error(f"Consistency evaluation failed: {e}")
            return {"error": str(e)}

    async def _evaluate_retrieval_quality(self) -> Dict[str, Any]:
        """Evaluate retrieval system quality"""
        try:
            # Test different types of queries
            query_types = {
                "character_specific": [
                    "Byleth's interactions with Dimitri",
                    "Annette's feelings about Sylvain",
                    "Dedue's observations about workplace relationships"
                ],
                "relationship_focused": [
                    "romantic relationships in the office",
                    "secret affairs and hidden meetings",
                    "trust and betrayal between characters"
                ],
                "temporal": [
                    "events in early chapters",
                    "relationship development over time",
                    "changes in character dynamics"
                ],
                "emotional": [
                    "feelings of suspicion and doubt",
                    "moments of happiness and joy",
                    "anger and confrontation"
                ]
            }

            quality_metrics = {}

            for query_type, queries in query_types.items():
                type_results = []

                for query_text in queries:
                    memory_query = MemoryQuery(
                        query_text=query_text,
                        limit=10
                    )

                    response = await self.retriever.search_memories(memory_query)

                    # Analyze quality metrics
                    avg_similarity = sum(r.similarity_score for r in response.results) / len(response.results) if response.results else 0
                    retrieval_methods = [r.retrieval_method for r in response.results]
                    method_diversity = len(set(retrieval_methods))

                    type_results.append({
                        "query": query_text,
                        "results_count": len(response.results),
                        "avg_similarity": avg_similarity,
                        "method_diversity": method_diversity,
                        "methods_used": list(set(retrieval_methods))
                    })

                quality_metrics[query_type] = {
                    "queries": type_results,
                    "avg_results_count": sum(r["results_count"] for r in type_results) / len(type_results),
                    "avg_similarity": sum(r["avg_similarity"] for r in type_results) / len(type_results)
                }

            return quality_metrics

        except Exception as e:
            logger.error(f"Retrieval quality evaluation failed: {e}")
            return {"error": str(e)}

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics"""
        try:
            async with self.db_manager.get_connection() as conn:
                # Basic counts
                total_memories = await conn.fetchval("SELECT COUNT(*) FROM memories")
                total_characters = await conn.fetchval("SELECT COUNT(*) FROM characters")
                total_users = await conn.fetchval("SELECT COUNT(*) FROM users")

                # Memory distribution
                memory_types = await conn.fetch(
                    "SELECT memory_type, COUNT(*) as count FROM memories GROUP BY memory_type"
                )

                # Chapter coverage
                chapter_stats = await conn.fetch("""
                    SELECT
                        MIN(chapter_number) as min_chapter,
                        MAX(chapter_number) as max_chapter,
                        COUNT(DISTINCT chapter_number) as unique_chapters,
                        AVG(CAST(chapter_number AS FLOAT)) as avg_chapter
                    FROM memories
                    WHERE chapter_number IS NOT NULL
                """)

                # Access patterns
                access_stats = await conn.fetch("""
                    SELECT
                        AVG(access_count) as avg_access_count,
                        MAX(access_count) as max_access_count,
                        COUNT(*) FILTER (WHERE access_count = 0) as never_accessed
                    FROM memories
                """)

                # Importance distribution
                importance_stats = await conn.fetch("""
                    SELECT
                        AVG(importance_score) as avg_importance,
                        MIN(importance_score) as min_importance,
                        MAX(importance_score) as max_importance,
                        COUNT(*) FILTER (WHERE importance_score >= 7.0) as high_importance,
                        COUNT(*) FILTER (WHERE importance_score >= 4.0 AND importance_score < 7.0) as medium_importance,
                        COUNT(*) FILTER (WHERE importance_score < 4.0) as low_importance
                    FROM memories
                """)

                return {
                    "total_memories": total_memories,
                    "total_characters": total_characters,
                    "total_users": total_users,
                    "memory_type_distribution": {row['memory_type']: row['count'] for row in memory_types},
                    "chapter_coverage": dict(chapter_stats[0]) if chapter_stats else {},
                    "access_patterns": dict(access_stats[0]) if access_stats else {},
                    "importance_distribution": dict(importance_stats[0]) if importance_stats else {}
                }

        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}

    async def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries for evaluation"""
        # In a real implementation, this would load from eval/queries.jsonl
        # For now, we'll return hardcoded test queries based on the narrative
        return [
            {
                "query": "Byleth's romantic relationship with Dimitri",
                "expected_ids": [],  # Would be populated with actual memory IDs
                "description": "Should find memories about intimate moments between Byleth and Dimitri"
            },
            {
                "query": "Sylvain's affair and secret meetings",
                "expected_ids": [],
                "description": "Should find memories about Sylvain's secret relationship with Byleth"
            },
            {
                "query": "Annette's surprise plans for Sylvain",
                "expected_ids": [],
                "description": "Should find memories about Annette planning romantic getaway"
            },
            {
                "query": "Dedue's suspicious observations",
                "expected_ids": [],
                "description": "Should find memories about Dedue discovering evidence of affairs"
            },
            {
                "query": "Office dynamics and professional relationships",
                "expected_ids": [],
                "description": "Should find memories about workplace interactions"
            }
        ]

    async def _get_total_memory_count(self) -> int:
        """Get total number of memories in the system"""
        try:
            async with self.db_manager.get_connection() as conn:
                return await conn.fetchval("SELECT COUNT(*) FROM memories")
        except Exception:
            return 0

    async def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Evaluation results saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")


async def main():
    """Main function to run evaluation"""
    from app.deps import DatabaseManager, EmbeddingService

    # Initialize services
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()

    try:
        # Initialize database connection
        await db_manager.init_pool()

        # Create evaluation runner
        runner = EvaluationRunner(db_manager, embedding_service)

        # Run full evaluation
        results = await runner.run_full_evaluation()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"eval/results/evaluation_{timestamp}.json"
        await runner.save_results(results, output_path)

        print(f"Evaluation completed. Results saved to {output_path}")

        # Print summary
        print("\n=== Evaluation Summary ===")
        if "precision_recall" in results:
            pr = results["precision_recall"]
            print(f"Average Precision@5: {pr.get('average_precision_at_k', {}).get(5, 0):.3f}")
            print(f"Average Recall@5: {pr.get('average_recall_at_k', {}).get(5, 0):.3f}")
            print(f"Average MRR: {pr.get('average_mrr', 0):.3f}")

        if "consistency" in results:
            cons = results["consistency"]
            print(f"Overall Consistency Score: {cons.get('overall_consistency_score', 0):.3f}")
            print(f"Total Issues Found: {cons.get('total_issues', 0)}")

        if "system_metrics" in results:
            metrics = results["system_metrics"]
            print(f"Total Memories: {metrics.get('total_memories', 0)}")
            print(f"Total Characters: {metrics.get('total_characters', 0)}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Evaluation failed: {e}")

    finally:
        await db_manager.close_pool()


if __name__ == "__main__":
    asyncio.run(main())