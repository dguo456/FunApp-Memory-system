#!/usr/bin/env python3
"""
Database seeding script to initialize the database with characters and basic data.
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
    from app.deps import DatabaseManager, setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run this script from the sekai-memory directory")
    print("Or install the package in development mode: pip install -e .")
    sys.exit(1)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


async def seed_database():
    """Seed the database with initial data"""

    db_manager = DatabaseManager()

    try:
        # Initialize database connection
        await db_manager.init_pool()
        logger.info("Database connection established")

        async with db_manager.get_connection() as conn:
            # Check if data already exists
            existing_chars = await conn.fetchval("SELECT COUNT(*) FROM characters")
            if existing_chars > 0:
                logger.info(f"Database already has {existing_chars} characters. Skipping seeding.")
                print(f"Database already seeded with {existing_chars} characters.")
                return

            logger.info("Seeding database with initial data...")

            # Insert characters from the narrative
            characters = [
                ('Byleth', 'Strategic and calculating main character who orchestrates complex relationships'),
                ('Dimitri', 'Intense and focused executive who becomes romantically involved with Byleth'),
                ('Sylvain', 'Charming and flirtatious character who gets entangled in an affair'),
                ('Annette', 'Cheerful and trusting character in a relationship with Sylvain'),
                ('Felix', 'Sharp and analytical character who observes others with suspicion'),
                ('Dedue', 'Loyal and observant character who discovers evidence of workplace affairs'),
                ('Mercedes', 'Kind and supportive character who provides emotional support'),
                ('Ashe', 'Gentle and concerned character who worries about workplace dynamics')
            ]

            for name, description in characters:
                await conn.execute(
                    "INSERT INTO characters (name, description) VALUES ($1, $2)",
                    name, description
                )

            logger.info(f"Inserted {len(characters)} characters")

            # Insert default user
            await conn.execute(
                "INSERT INTO users (name) VALUES ($1)",
                "User"
            )

            logger.info("Inserted default user")

            # Insert initial world state
            await conn.execute(
                "INSERT INTO world_states (chapter_number, state_description) VALUES ($1, $2)",
                1, "Normal corporate office environment at Garreg Mach Corp. Professional relationships and workplace dynamics are established."
            )

            logger.info("Inserted initial world state")

            # Verify the data
            char_count = await conn.fetchval("SELECT COUNT(*) FROM characters")
            user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
            world_count = await conn.fetchval("SELECT COUNT(*) FROM world_states")

            print(f"\n=== Database Seeding Complete ===")
            print(f"Characters created: {char_count}")
            print(f"Users created: {user_count}")
            print(f"World states created: {world_count}")

            # List all characters
            chars = await conn.fetch("SELECT id, name, description FROM characters ORDER BY name")
            print(f"\n=== Characters ===")
            for char in chars:
                print(f"ID {char['id']}: {char['name']} - {char['description'][:50]}...")

            logger.info("Database seeding completed successfully")

    except Exception as e:
        logger.error(f"Database seeding failed: {e}")
        print(f"Error seeding database: {e}")

    finally:
        await db_manager.close_pool()
        logger.info("Database connection closed")


async def reset_database():
    """Reset the database by clearing all data"""

    db_manager = DatabaseManager()

    try:
        await db_manager.init_pool()
        logger.info("Database connection established")

        async with db_manager.get_connection() as conn:
            # Delete all data in reverse dependency order
            tables = [
                'consistency_ledger',
                'memory_relationships',
                'memories',
                'world_states',
                'users',
                'characters'
            ]

            for table in tables:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                if count > 0:
                    await conn.execute(f"DELETE FROM {table}")
                    logger.info(f"Cleared {count} records from {table}")

            # Reset sequences
            sequences = [
                'characters_id_seq',
                'users_id_seq',
                'memories_id_seq',
                'memory_relationships_id_seq',
                'consistency_ledger_id_seq',
                'world_states_id_seq'
            ]

            for seq in sequences:
                await conn.execute(f"ALTER SEQUENCE {seq} RESTART WITH 1")

            print("Database reset complete. All data cleared.")
            logger.info("Database reset completed")

    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        print(f"Error resetting database: {e}")

    finally:
        await db_manager.close_pool()


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Database seeding utility")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database by clearing all data before seeding"
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Only reset database, don't seed new data"
    )

    args = parser.parse_args()

    if args.reset_only:
        await reset_database()
    elif args.reset:
        await reset_database()
        await seed_database()
    else:
        await seed_database()


if __name__ == "__main__":
    asyncio.run(main())