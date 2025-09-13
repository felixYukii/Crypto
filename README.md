# Cryptocurrency Transaction Analysis & Forensics

## Project Overview
This project provides tools for analyzing cryptocurrency transactions and detecting suspicious activities. It consists of two main parts:

1. **Bitcoin Transaction Analysis**: Visualizes transaction graphs and flow patterns for a specific Bitcoin address
2. **SHIB Token Forensics**: Analyzes token transfers to detect anomalous activities and potential market manipulation

## Installation

1. Clone the repository:
git clone https://github.com/yourusername/crypto-forensics-analysis.git
cd crypto

2. Install dependencies:
pip install -r requirements.txt

### Example Usage
Part 1: Bitcoin Transaction Analysis

python src/part1.py

This will:
Fetch transaction data for address bc1qmnvc7gv9ydrq5fssl072qzd60rk3hcqzmcjtmz
Generate a transaction graph visualization
Create a Bitcoin flow analysis chart

Part 2: SHIB Token Forensics

python src/part2.py

This will:
Fetch SHIB token transfer data from Etherscan
Analyze transactions between August 15-September 15, 2020
Detect anomalous activities using statistical methods
Generate visualizations and CSV reports

### Configuration
For Part 2, you need to:

1. Obtain an API key from Etherscan
2. Replace the placeholder API key in part2.py:

ETHERSCAN_API_KEY = "YOUR_ACTUAL_API_KEY"