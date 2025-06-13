# test_env.py
from dotenv import load_dotenv
import os

load_dotenv()
print("ELASTIC_INDEX =", os.getenv("ELASTIC_INDEX"))