import requests

#Part 1 API
def get_address_data():
    url = "https://blockchain.info/rawaddr/bc1qmnvc7gv9ydrq5fssl072qzd60rk3hcqzmcjtmz?limit=10"
    response = requests.get(url)
    response.raise_for_status()  # Error handling
    return response.json()

print(get_address_data())