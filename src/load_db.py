"""Load the cleaned dataset into the database:  python -m src.load_db"""
from src.db import DATABASE_URL, load_patients
from src.preprocess import load_data

if __name__ == "__main__":
    n = load_patients(load_data())
    print(f"Loaded {n} patients into {DATABASE_URL}")
