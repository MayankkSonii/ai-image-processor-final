import os
import sys

# Ensure /app is in path
sys.path.append("/app")

import redis
from rq import Worker, Queue, Connection
from app.core.config import settings
from app.utils.logging_utils import setup_logger

logger = setup_logger("worker")

listen = ['default']
redis_url = settings.REDIS_URL

if __name__ == '__main__':
    logger.info(f"Worker starting. Listening on queues: {listen}")
    try:
        conn = redis.from_url(redis_url)
        with Connection(conn):
            worker = Worker(list(map(Queue, listen)))
            logger.info("Worker connection established. Ready for jobs.")
            worker.work()
    except Exception as e:
        logger.error(f"Worker failed to start or lost connection: {e}")
        sys.exit(1)
