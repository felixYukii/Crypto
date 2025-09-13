import requests
import matplotlib.pyplot as plt
import networkx as nx
import datetime
import matplotlib.dates as mdates
import os  # Added import

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

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
    plt.savefig('outputs/part1_transaction_graph.png', dpi=300, bbox_inches='tight')  # Changed to save
    plt.close()  # Added to close the figure

# 5. Additional bar graph for analysis
def plot_transaction_amounts(transactions):
    # Sort transactions by time (oldest to newest)
    transactions.sort(key=lambda x: x['time'])
    
    # Prepare data for plotting
    hashes = [tx['hash'][:6] + '...' + tx['hash'][-4:] for tx in transactions]  # More readable format
    values = [tx['value'] for tx in transactions]
    dates = [tx['time'] for tx in transactions]
    
    # Create colors based on sent vs received
    colors = ['red' if val < 0 else 'green' for val in values]
    
    # Create the plot with larger figure size
    plt.figure(figsize=(16, 10))
    bars = plt.bar(range(len(transactions)), values, color=colors, alpha=0.7)
    
    # Customize the plot with larger fonts
    plt.title('Bitcoin Amount per Transaction', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Transaction Hash', fontsize=16, labelpad=15)
    plt.ylabel('Bitcoin Amount', fontsize=16, labelpad=15)
    
    # Rotate x-axis labels 90 degrees for better readability
    plt.xticks(range(len(transactions)), hashes, rotation=90, ha='center', fontsize=12)
    
    # Increase y-axis label size
    plt.yticks(fontsize=12)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars with larger font
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        va = 'bottom' if value >= 0 else 'top'
        y_offset = 0.01 if value >= 0 else -0.01
        plt.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                 f'{value:.4f}', ha='center', va=va, fontsize=11, rotation=0)
    
    # Add a legend with larger font
    plt.plot([], [], 's', color='red', label='Sent BTC', markersize=10)
    plt.plot([], [], 's', color='green', label='Received BTC', markersize=10)
    plt.legend(loc='upper right', fontsize=14)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Add date information to the plot with larger font
    date_range = f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"
    plt.figtext(0.5, 0.01, f"Transaction Date Range: {date_range}", 
                ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.2, "pad":10})
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('outputs/part1_bitcoin_flow_analysis.png', dpi=300, bbox_inches='tight')  # Changed to save
    plt.close()  # Added to close the figure
    
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

    # 3. Create graph
    print("Building graph...")
    G = create_transaction_graph(transactions, focus_address)

    # 4. Plot graph
    print("Visualizing graph...")
    plot_graph(G)

    # 5. Plot Bitcoin flow analysis
    print("Creating Bitcoin flow analysis...")
    plot_transaction_amounts(transactions)

    print("Analysis complete! Check the outputs/ directory for results.")