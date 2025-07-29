import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.database import create_tables

def main():
    try:
        create_tables()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
