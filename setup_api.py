#!/usr/bin/env python3
"""
CHIBOY BOT - API Key Setup
===========================
Guide to setting up API keys for trading.
"""

import os
import sys

def print_header():
    print("=" * 60)
    print("   CHIBOY BOT - API Key Setup")
    print("=" * 60)
    print()

def get_oanda_keys():
    print("\n📊 OANDA (Forex Trading)")
    print("-" * 40)
    print("1. Go to: https://www.oanda.com/")
    print("2. Create a free demo account or login")
    print("3. Go to: My Account → Manage API Access")
    print("4. Generate your API key")
    print("5. Your Account ID is in your dashboard")
    print()
    api_key = input("Enter OANDA API Key: ").strip()
    account_id = input("Enter OANDA Account ID: ").strip()
    return api_key, account_id

def get_binance_keys():
    print("\n₿ BINANCE (Crypto Trading)")
    print("-" * 40)
    print("1. Go to: https://www.binance.com/")
    print("2. Create account (or use testnet)")
    print("3. Go to: Account → API Management")
    print("4. Create new API key")
    print("5. Enable futures for testnet trading")
    print()
    api_key = input("Enter Binance API Key: ").strip()
    secret_key = input("Enter Binance Secret Key: ").strip()
    return api_key, secret_key

def save_env_file(oanda_key, oanda_id, binance_key, binance_secret):
    env_content = f"""# CHIBOY BOT - API Configuration
# Generated automatically

# OANDA API (Forex)
OANDA_API_KEY={oanda_key}
OANDA_ACCOUNT_ID={oanda_id}
OANDA_ENV=practice

# Binance API (Crypto)
BINANCE_API_KEY={binance_key}
BINANCE_SECRET_KEY={binance_secret}
BINANCE_TESTNET=true

# Trading Mode
DRY_RUN=true
EXECUTE_TRADES=false

# Logging
LOG_LEVEL=INFO
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("\n✅ Configuration saved to .env file!")

def main():
    print_header()
    
    print("This wizard will help you configure API keys.")
    print("You can skip any step if you don't need that exchange.")
    print()
    
    # OANDA
    oanda_key = ""
    oanda_id = ""
    setup_oanda = input("Setup OANDA (Forex)? (y/n): ").strip().lower() == 'y'
    if setup_oanda:
        oanda_key, oanda_id = get_oanda_keys()
    
    # Binance
    binance_key = ""
    binance_secret = ""
    setup_binance = input("Setup Binance (Crypto)? (y/n): ").strip().lower() == 'y'
    if setup_binance:
        binance_key, binance_secret = get_binance_keys()
    
    # Save
    if oanda_key or binance_key:
        save_env_file(oanda_key, oanda_id, binance_key, binance_secret)
    else:
        print("\n⚠️ No API keys configured.")
        print("You can run in simulation mode without keys.")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
