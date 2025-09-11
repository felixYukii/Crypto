import requests

def get_address_data(address):
    url = f"https://blockchain.info/rawaddr/{address}?limit=10"
    response = requests.get(url)
    return response.json()

print(get_address_data("bc1qmnvc7gv9ydrq5fssl072qzd60rk3hcqzmcjtmz"))