import sqlite3
import json
from datetime import datetime

def init_database():
    """Initialize the fraud cases database with sample data"""
    conn = sqlite3.connect('fraud_cases.db')
    cursor = conn.cursor()
    
    # Create fraud_cases table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fraud_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userName TEXT NOT NULL,
            securityIdentifier TEXT NOT NULL,
            cardEnding TEXT NOT NULL,
            status TEXT DEFAULT 'pending_review',
            transactionName TEXT NOT NULL,
            transactionAmount REAL NOT NULL,
            transactionTime TEXT NOT NULL,
            transactionCategory TEXT NOT NULL,
            transactionSource TEXT NOT NULL,
            transactionLocation TEXT NOT NULL,
            securityQuestion TEXT NOT NULL,
            securityAnswer TEXT NOT NULL,
            outcome TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Sample fraud cases
    sample_cases = [
        {
            "userName": "John",
            "securityIdentifier": "12345",
            "cardEnding": "4242",
            "status": "pending_review",
            "transactionName": "ABC Industry",
            "transactionAmount": 1249.99,
            "transactionTime": "2025-11-27 02:34:15",
            "transactionCategory": "e-commerce",
            "transactionSource": "alibaba.com",
            "transactionLocation": "Shanghai, China",
            "securityQuestion": "What is your mother's maiden name?",
            "securityAnswer": "Smith"
        },
        {
            "userName": "Sarah",
            "securityIdentifier": "67890",
            "cardEnding": "8888",
            "status": "pending_review",
            "transactionName": "Luxury Watches Inc",
            "transactionAmount": 3599.00,
            "transactionTime": "2025-11-26 18:22:41",
            "transactionCategory": "retail",
            "transactionSource": "luxurywatches.example",
            "transactionLocation": "Geneva, Switzerland",
            "securityQuestion": "What city were you born in?",
            "securityAnswer": "Boston"
        },
        {
            "userName": "Michael",
            "securityIdentifier": "54321",
            "cardEnding": "1111",
            "status": "pending_review",
            "transactionName": "Gaming Store XYZ",
            "transactionAmount": 899.50,
            "transactionTime": "2025-11-27 09:15:30",
            "transactionCategory": "gaming",
            "transactionSource": "gamingstore.example",
            "transactionLocation": "Tokyo, Japan",
            "securityQuestion": "What is your favorite color?",
            "securityAnswer": "Blue"
        }
    ]
    
    # Insert sample cases if table is empty
    cursor.execute("SELECT COUNT(*) FROM fraud_cases")
    if cursor.fetchone()[0] == 0:
        for case in sample_cases:
            cursor.execute('''
                INSERT INTO fraud_cases 
                (userName, securityIdentifier, cardEnding, status, transactionName, 
                 transactionAmount, transactionTime, transactionCategory, transactionSource,
                 transactionLocation, securityQuestion, securityAnswer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                case["userName"],
                case["securityIdentifier"],
                case["cardEnding"],
                case["status"],
                case["transactionName"],
                case["transactionAmount"],
                case["transactionTime"],
                case["transactionCategory"],
                case["transactionSource"],
                case["transactionLocation"],
                case["securityQuestion"],
                case["securityAnswer"]
            ))
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

def get_fraud_case_by_username(username):
    """Retrieve a pending fraud case for a specific username"""
    conn = sqlite3.connect('fraud_cases.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM fraud_cases 
        WHERE userName = ? AND status = 'pending_review'
        LIMIT 1
    ''', (username,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))
    return None

def update_fraud_case(case_id, status, outcome):
    """Update the fraud case status and outcome"""
    conn = sqlite3.connect('fraud_cases.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE fraud_cases 
        SET status = ?, outcome = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (status, outcome, case_id))
    
    conn.commit()
    conn.close()
    print(f"Case {case_id} updated: {status} - {outcome}")

if __name__ == "__main__":
    init_database()
    print("\nSample fraud cases created:")
    print("- Username: John (Security ID: 12345)")
    print("- Username: Sarah (Security ID: 67890)")
    print("- Username: Michael (Security ID: 54321)")