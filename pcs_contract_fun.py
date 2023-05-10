from web3 import Web3
import json
import datetime
from get_oracel_bnb_price import get_bnb_usd_price
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

# Replace this with the actual contract address of the smart contract
contract_address = '0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA'


# Load the ABI from a JSON file
with open('./json/pcs_abi.json', 'r') as f:
    abi_json = f.read()

# Convert the JSON ABI to a Python list
abi = json.loads(abi_json)

# Create a contract instance
contract = w3.eth.contract(address=contract_address, abi=abi)

def format_float(x):
    return float("{:.2f}".format(x))

def convert_to_bnb(amount):
    
    bnb_decimal = 10 ** 18
    bnb_value = amount / bnb_decimal
    return format_float(bnb_value)



@retry(wait=wait_fixed(300), stop=stop_after_attempt(15))
def get_round_data_round_with_titles(round_id):
    try:
        time.sleep(1)
        round_data = contract.functions.rounds(round_id).call()
        lock_price = round_data[4] / 10**8
        close_price = round_data[5] / 10**8
        pattern = 'Bull' if close_price > lock_price else 'Bear'
        bull_amount = int(convert_to_bnb(round_data[9]))
        bear_amount = int(convert_to_bnb(round_data[10]))
        trading_volume = bull_amount + bear_amount

        return {
            'Epoch': round_data[0],
            'Start Timestamp': round_data[1],
            'Lock Timestamp': round_data[2],
            'Close Timestamp': round_data[3],
            'Lock Price': format_float(lock_price),
            'Close Price': format_float(close_price),
            'Lock Oracle ID': round_data[6],
            'Close Oracle ID': round_data[7],
            'Total Amount': convert_to_bnb(round_data[8]),
            'Bull Amount': bull_amount,
            'Bear Amount': bear_amount,
            'Reward Base Cal Amount': convert_to_bnb(round_data[11]),
            'Reward Amount': convert_to_bnb(round_data[12]),
            'Oracle Called': round_data[13],
            'Pattern': pattern,
            'Latest Price': format_float(get_bnb_usd_price()),
            # 'Trading Volume': convert_to_bnb(round_data[8]),
            'Trading Volume': trading_volume,
        }
    except Exception as e:
        print(f"Error occurred: {e}")
        # Handle the error as desired, e.g. return a default value or an empty dict
        raise


def current_epoch():
    # Get the current epoch
    return contract.functions.currentEpoch().call()