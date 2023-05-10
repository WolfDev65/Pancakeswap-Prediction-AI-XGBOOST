from web3 import Web3
import json
import time
from tenacity import retry, wait_fixed, stop_after_attempt
import random

url_list = [
    "https://bsc-dataseed1.binance.org/",
    "https://bsc-dataseed2.binance.org/",
    "https://bsc-dataseed3.binance.org/",
    "https://bsc-dataseed4.binance.org/",
    "https://bsc-dataseed1.defibit.io/",
    "https://bsc-dataseed2.defibit.io/",
    "https://bsc-dataseed3.defibit.io/",
    "https://bsc-dataseed4.defibit.io/",
    "https://bsc-dataseed1.ninicoin.io/",
    "https://bsc-dataseed2.ninicoin.io/",
    "https://bsc-dataseed3.ninicoin.io/",
    "https://bsc-dataseed4.ninicoin.io/"
]


# Connect to the Binance Smart Chain network
@retry(wait=wait_fixed(300), stop=stop_after_attempt(15))
def connect_to_provider():
    return Web3(Web3.HTTPProvider(str(random.choice(url_list))))

w3 = connect_to_provider()

# Define a function to get the latest BNB/USD price from Chainlink
@retry(wait=wait_fixed(300), stop=stop_after_attempt(15))
def get_bnb_usd_price():
    try:
        time.sleep(1)
        # Replace this with the actual contract address of the Chainlink BNB/USD Price Feed
        contract_address = '0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE'

        # Load the ABI from a JSON file
        with open('./json/oracel_bnb_abi.json', 'r') as f:
            abi_json = f.read()

        # Convert the JSON ABI to a Python list
        abi = json.loads(abi_json)

        # Create a contract instance
        contract = w3.eth.contract(address=contract_address, abi=abi)

        # Get the latest BNB/USD price from Chainlink
        latest_price = contract.functions.latestAnswer().call()

        # Convert the latest price from wei to USD
        bnb_usd_price = latest_price / 10**8

        # Return the BNB/USD price in USD
        return bnb_usd_price

    except Exception as e:
        print(f"Error occurred: {e}")
        # Handle the error as desired, e.g. return a default value or None
        raise
