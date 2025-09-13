# Cryptocurrency Transaction Analysis & Forensics

## Project Overview
This project provides tools for analyzing cryptocurrency transactions and detecting suspicious activities. It consists of two main parts:

1. **Bitcoin Transaction Analysis**: Visualizes transaction graphs and flow patterns for a specific Bitcoin address
2. **SHIB Token Forensics**: Analyzes token transfers to detect anomalous activities and potential market manipulation

## Installation

1. Clone the repository:
git clone https://github.com/felixyukii/crypto.git

2. Install dependencies:
pip install -r requirements.txt

### Example Usage
Part 1: Bitcoin Transaction Analysis

python src/part1.py

This will:
Fetch transaction data for address bc1qmnvc7gv9ydrq5fssl072qzd60rk3hcqzmcjtmz
Generate a transaction graph visualization
Create a Bitcoin flow analysis chart
View result in /outputs

Part 2: SHIB Token Forensics

python src/part2.py

This will:
Fetch SHIB token transfer data from Etherscan
Analyze transactions between August 15 to September 15, 2020
Detect anomalous activities using statistical methods
Generate visualizations and CSV reports
View result in /outputs/part2_shib_analysis

### Configuration
For Part 2, you need to:

1. Obtain an API key from Etherscan
2. Replace the placeholder API key in part2.py:

ETHERSCAN_API_KEY = "YOUR_ACTUAL_API_KEY"

### Results

Part 1:
<img width="3660" height="2537" alt="part1_transaction_graph" src="https://github.com/user-attachments/assets/742fb132-69d0-4536-ad20-e874469ad75c" />
<img width="4776" height="2844" alt="part1_bitcoin_flow_analysis" src="https://github.com/user-attachments/assets/5861467c-b02b-4944-aa9c-7c2bcfa06ea4" />

Transaction f933332466f954757c84e97cada98df923dfa2196c7a065de17c3282b0a08fb8 (2022-08-09)
Type: Outgoing

Value: -141.72472834 BTC

Pattern: CoinJoin

Description:
The substantial transfer to an unfamiliar address is a significant warning signal, especially when there were multiple inputs. This pattern obscures the provenance of funds, likely through the use of CoinJoin. 

The receiving address likely functions as an intermediary, consolidating assets from multiple users to break the observable link between source and destination addresses. The objective is to return BTC (minus fees) originating from unrelated sources, thus effectively severing ties to the original sender. 

Many illict activities uses this level of obfuscation, so this activity is suspicious.

Part 2:

<img width="4800" height="2100" alt="shib_daily_volume_anomalies_annotated" src="https://github.com/user-attachments/assets/6291b646-6a55-48e1-b8a9-cbeb0d86ac60" />
There's a spike around September 4th marked as "Suspicious: z>3.0" with extremely high on-chain volume (300+ trillion tokens). This represents a statistical anomaly, over 3 standard deviations from normal activity. One common reasons is it's a pump-and-dump scheme.\

<img width="1200" height="600" alt="shib_intertx_time_histogram" src="https://github.com/user-attachments/assets/72451cc1-f648-4c87-9547-0d77770674de" />
Histogram shows suspicious timing behaviors. While most transactions occur within seconds (the high left peak), there's an unusual spike at exactly 3600 seconds (1 hour), suggesting automation. This could indicate bot networks.\

<img width="1800" height="1237" alt="shib_wallet_hour_heatmap" src="https://github.com/user-attachments/assets/03689280-41b4-4065-9466-1cc1033bafe1" />
The heatmap reveals highly suspicious patterns where wallet 0x811beed011 shows intense activity. This could be due to automated bot operations, coordinated pump-and-dump schemes and market manipulation campaigns.
