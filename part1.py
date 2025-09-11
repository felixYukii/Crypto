import requests
import matplotlib.pyplot as plt
import networkx as nx
import datetime

# 1. Fetch address data
def get_address_data():
    url = "https://blockchain.info/rawaddr/bc1qmnvc7gv9ydrq5fssl072qzd60rk3hcqzmcjtmz?limit=10"
    response = requests.get(url)
    response.raise_for_status()  # Error handling
    return response.json()

# 2. Process transactions
def process_transactions(address_data, focus_address):
    transactions = []
    
    for tx in address_data['txs']:
        inputs = []
        outputs = []
        involves_focus = False  # Track if this tx includes our focus address
        
        # --- Process inputs (senders) ---
        for input in tx['inputs']:
            if 'prev_out' in input and 'addr' in input['prev_out']:
                addr = input['prev_out']['addr']
                inputs.append(addr)
                if addr == focus_address:
                    involves_focus = True
        
        # --- Process outputs (receivers) ---
        for output in tx['out']:
            if 'addr' in output:
                addr = output['addr']
                outputs.append(addr)
                if addr == focus_address:
                    involves_focus = True
        
        # --- Add only relevant transactions ---
        if involves_focus:
            transactions.append({
                'hash': tx['hash'],
                'inputs': inputs,
                'outputs': outputs,
                'time': datetime.datetime.fromtimestamp(tx['time']),  
                'value': tx['result'] / 100000000  
            })
    
    return transactions

# 3. Create transaction graph
def create_transaction_graph(transactions, focus_address):
    """
    Build a directed graph from the list of transactions
    """
    G = nx.DiGraph()
    
    # Add focus address as a central node
    G.add_node(focus_address, size=1000, color='red')
    
    # Process transactions
    for tx in transactions:
        # If focus address is sending BTC
        if focus_address in tx['inputs']:
            for output in tx['outputs']:
                if output != focus_address:  # Avoid self-loops
                    if G.has_edge(focus_address, output):
                        G[focus_address][output]['weight'] += 1
                    else:
                        G.add_edge(focus_address, output, weight=1)
                        G.nodes[output]['color'] = 'lightblue'
        
        # If focus address is receiving BTC
        if focus_address in tx['outputs']:
            for input_addr in tx['inputs']:
                if input_addr != focus_address:
                    if G.has_edge(input_addr, focus_address):
                        G[input_addr][focus_address]['weight'] += 1
                    else:
                        G.add_edge(input_addr, focus_address, weight=1)
                        G.nodes[input_addr]['color'] = 'lightgreen'
    
    return G

# 4. Plot the graph
def plot_graph(G):
    pos = nx.spring_layout(G, k=0.5, seed=42)  # Positions for nodes

    # Node attributes
    node_colors = [G.nodes[node].get('color', 'gray') for node in G.nodes()]
    node_sizes = [G.nodes[node].get('size', 500) for node in G.nodes()]

    # Edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Draw graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color='gray',
            width=edge_weights,
            font_size=8)

    plt.title("Transaction Graph", fontsize=16)
    plt.show()

# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    focus_address = "bc1qmnvc7gv9ydrq5fssl072qzd60rk3hcqzmcjtmz"
    
    # 1. Get data
    print("Fetching blockchain data...")
    address_data = get_address_data()

    # 2. Process transactions
    print("Processing transactions...")
    transactions = process_transactions(address_data, focus_address)
    print(transactions)

    # 3. Create graph
    print("Building graph...")
    G = create_transaction_graph(transactions, focus_address)

    # 4. Plot graph
    print("Visualizing graph...")
    plot_graph(G)

    print("Analysis complete!")
