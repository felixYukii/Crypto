import requests

# Part 2 API
# Add your Etherscan API key at the end of the URL
# E.g.  "https://api.etherscan.io/api?module=account&action=tokentx&contractaddress=0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE&page=1&offset=10&sort=asc&apikey=5QKZSIUQ5218WBDDBEYVANYS8SXF6Y4N8"

def get_address_data():
    url = (
        "https://api.etherscan.io/api?module=account&action=tokentx&contractaddress=0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE&page=1&offset=10&sort=asc&apikey="
   )
    response = requests.get(url)
    response.raise_for_status()  # Error handling
    return response.json()

print(get_address_data())