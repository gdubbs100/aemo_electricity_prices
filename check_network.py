import socket
import requests
import aiohttp
import os

import sys
import asyncio

# FORCE Windows to use the standard Selector loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def check_all():
    host = "api.openelectricity.org.au"
    
    print(f"--- 1. Basic DNS Lookup ---")
    try:
        ip = socket.gethostbyname(host)
        print(f"✅ Success: {host} is at {ip}")
    except socket.gaierror as e:
        print(f"❌ DNS Failed: Could not resolve {host}. (System level issue)")

    print(f"\n--- 2. Environment Variables ---")
    # Checking for proxies that might be confusing aiohttp
    proxies = {k: v for k, v in os.environ.items() if "PROXY" in k.upper()}
    if proxies:
        print(f"⚠️ Found proxy variables: {proxies}")
    else:
        print("✅ No proxy variables detected.")

    print(f"\n--- 3. Synchronous HTTP (Requests) ---")
    try:
        response = requests.get(f"https://{host}/v1/facilities", timeout=5)
        print(f"✅ Success: Status Code {response.status_code}")
    except Exception as e:
        print(f"❌ Requests Failed: {type(e).__name__}")

    print(f"\n--- 4. Asynchronous HTTP (aiohttp) ---")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://{host}/v1/facilities", timeout=5) as resp:
                print(f"✅ Success: Status Code {resp.status}")
    except Exception as e:
        print(f"❌ aiohttp Failed: {e}")

if __name__ == "__main__":
    asyncio.run(check_all())