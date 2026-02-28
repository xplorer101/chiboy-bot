"""
Database module for CHIBOY BOT
Handles SQLite database for users and signals
"""

import sqlite3
import hashlib
import os
from datetime import datetime
from contextlib import contextmanager

DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chiboy.db')


@contextmanager
def get_db_connection():
    """Get a database connection with row factory"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize database tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl REAL NOT NULL,
                tp REAL NOT NULL,
                confidence REAL,
                reasons TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_user_id ON signals(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at)')
        
        print("Database initialized successfully!")


def hash_password(password):
    """Hash a password using SHA-256 with salt"""
    salt = os.urandom(32).hex()
    pwd_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{pwd_hash}"


def verify_password(password, password_hash):
    """Verify a password against its hash"""
    try:
        salt, pwd_hash = password_hash.split(':')
        return pwd_hash == hashlib.sha256((salt + password).encode()).hexdigest()
    except:
        return False


def create_user(username, password):
    """Create a new user"""
    try:
        password_hash = hash_password(password)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (username, password_hash) VALUES (?, ?)',
                (username, password_hash)
            )
            return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None  # Username already exists


def get_user_by_username(username):
    """Get user by username"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        return cursor.fetchone()


def get_user_by_id(user_id):
    """Get user by ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        return cursor.fetchone()


def save_signal(user_id, symbol, timeframe, direction, entry_price, sl, tp, confidence=0, reasons=""):
    """Save a trading signal to the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO signals 
               (user_id, symbol, timeframe, direction, entry_price, sl, tp, confidence, reasons)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (user_id, symbol, timeframe, direction, entry_price, sl, tp, confidence, reasons)
        )
        return cursor.lastrowid


def get_signals(user_id=None, symbol=None, start_date=None, end_date=None, limit=100):
    """Get signals with optional filters"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        query = 'SELECT * FROM signals WHERE 1=1'
        params = []
        
        if user_id:
            query += ' AND user_id = ?'
            params.append(user_id)
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        if start_date:
            query += ' AND created_at >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND created_at <= ?'
            params.append(end_date)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        return cursor.fetchall()


# Initialize database on import
init_database()
