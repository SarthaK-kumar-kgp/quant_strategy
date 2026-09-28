import os
from dotenv import load_dotenv
from py_clob_client_v2 import ClobClient

load_dotenv()

HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet
PRIVATE_KEY = "PUT UR KEY"

# This derives (or creates) your API key/secret/passphrase
client = ClobClient(host=HOST, chain_id=CHAIN_ID, key=PRIVATE_KEY)
creds = client.create_or_derive_api_key()

print("API Key:", creds.api_key)
print("Secret:", creds.api_secret)
print("Passphrase:", creds.api_passphrase)