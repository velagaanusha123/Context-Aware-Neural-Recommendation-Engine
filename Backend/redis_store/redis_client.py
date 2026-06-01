import redis

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

try:
    redis_client.ping()
    print("[INFO] Redis Connected Successfully")
except Exception as e:
    print("[ERROR]", e)