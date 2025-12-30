import redis
from rq import Queue
from app.core.config import settings

# Redis connection
redis_conn = redis.from_url(settings.REDIS_URL)

# RQ Queue
queue = Queue('default', connection=redis_conn)
