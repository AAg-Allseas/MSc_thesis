import os

def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ