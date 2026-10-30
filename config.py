import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Redis
    REDIS_URL = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379/0",
    )

    # Rate limiting
    GLOBAL_LIMIT = int(os.getenv("GLOBAL_LIMIT", "100"))
    USER_LIMIT = int(os.getenv("USER_LIMIT", "10"))  # Instead of lim being diff stored in diff files
    # we store it in one place to configure; just like we do in .env. 
    RATE_LIMIT_WINDOW = int(
        os.getenv("RATE_LIMIT_WINDOW", "60")
    )

    # Input/output validation
    MAX_QUERY_LENGTH = 200
    MAX_ANSWER_LENGTH = 3000

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


settings = Settings()