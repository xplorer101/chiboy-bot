"""
Vercel Python API Handler for CHIBOY BOT
"""
import os
import sys

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.chdir(root)

from webapp import app

# Vercel handler - returns WSGI iterable
def handler(environ, start_response):
    """WSGI-compatible handler for Vercel"""
    return app(environ, start_response)
