#!/usr/bin/env python3
"""
Script to load chapter data from JSON file and process it through the memory system.
Points to /mnt/data/memory_data.json as specified in the requirements.
"""

import asyncio
import sys
import logging
import os
from pathlib import Path

# Add the current directory to the path so we can import our modules
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = str(current_dir)

try:
    from core.ingest import MemoryIngestionService
    from app.deps import DatabaseManager, EmbeddingService, setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run this script from the sekai-memory directory")
    print("Or install the package in development mode: pip install -e .")
    sys.exit(1)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


async def load_and_process_json(json_file_path: str, batch_size: int = 10):
    """Load JSON data and process it through the memory system"""

    # Initialize services
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()

    try:
        # Initialize database connection
        await db_manager.init_pool()
        logger.info("Database connection established")

        # Create ingestion service
        ingestion_service = MemoryIngestionService(db_manager)

        # Load chapter data from JSON
        logger.info(f"Loading chapter data from {json_file_path}")
        chapters = await ingestion_service.load_json_data(json_file_path)

        if not chapters:
            logger.error("No chapters loaded from JSON file")
            return

        logger.info(f"Loaded {len(chapters)} chapters")

        # Process all chapters
        logger.info("Starting chapter processing...")
        results = await ingestion_service.process_all_chapters(chapters, batch_size)

        # Print summary
        total_created = sum(r.memories_created for r in results)
        total_updated = sum(r.memories_updated for r in results)
        total_consistency_issues = sum(len(r.consistency_issues) for r in results)

        print("\n=== Processing Summary ===")
        print(f"Chapters processed: {len(results)}")
        print(f"Memories created: {total_created}")
        print(f"Memories updated: {total_updated}")
        print(f"Consistency issues found: {total_consistency_issues}")
        print(f"World states updated: {sum(1 for r in results if r.world_state_updated)}")

        # Print per-chapter breakdown
        print("\n=== Chapter Breakdown ===")
        for result in results:
            print(f"Chapter {result.chapter_number}: {result.memories_created} created, "
                  f"{result.memories_updated} updated, {len(result.consistency_issues)} issues")

        logger.info("Chapter processing completed successfully")

    except FileNotFoundError:
        logger.error(f"JSON file not found: {json_file_path}")
        print(f"Error: Could not find file {json_file_path}")
        print("Please ensure the file exists and the path is correct.")

    except Exception as e:
        logger.error(f"Failed to process JSON data: {e}")
        print(f"Error processing data: {e}")

    finally:
        await db_manager.close_pool()
        logger.info("Database connection closed")


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Load and process memory data from JSON")
    parser.add_argument(
        "--file",
        default="/mnt/data/memory_data.json",
        help="Path to JSON file (default: /mnt/data/memory_data.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing chapters (default: 10)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local memory_data.json file instead of /mnt/data/ path"
    )

    args = parser.parse_args()

    # Determine the correct file path
    if args.local:
        # Look for memory_data.json in the current directory or parent directories
        current_dir = Path.cwd()
        json_file = None

        # Check current directory and parent directories
        for path in [current_dir] + list(current_dir.parents):
            potential_file = path / "memory_data.json"
            if potential_file.exists():
                json_file = str(potential_file)
                break

        if not json_file:
            print("Error: Could not find memory_data.json in current directory or parent directories")
            print("Please run this script from the project root or specify --file path")
            sys.exit(1)

        print(f"Using local file: {json_file}")
    else:
        json_file = args.file
        print(f"Using specified file: {json_file}")

    # Run the processing
    await load_and_process_json(json_file, args.batch_size)


if __name__ == "__main__":
    asyncio.run(main())