import os
from dotenv import load_dotenv
from src.app import run_app

if __name__ == "__main__":
    load_dotenv()
    run_app()