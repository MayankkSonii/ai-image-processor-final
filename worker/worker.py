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
    import time
    conn = None
    retries = 10
    while retries > 0:
        try:
            logger.info(f"Connecting to Redis at {redis_url} (Retries left: {retries})...")
            conn = redis.from_url(redis_url)
            conn.ping()
            break
        except Exception as e:
            retries -= 1
            if retries == 0:
                logger.error(f"Could not connect to Redis after 10 attempts: {e}")
                sys.exit(1)
            time.sleep(3)

    logger.info(f"Worker starting. Listening on queues: {listen}")
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        logger.info("Worker connection established. Ready for jobs.")
        worker.work()
