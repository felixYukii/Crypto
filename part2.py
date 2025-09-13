"""
shib_forensics_full.py

Robust forensic script for the Shiba Inu (SHIB) ERC-20 token using Etherscan tokentx endpoint.
Features:
 - Respects Etherscan page * offset <= 10000 limit (uses PAGE_OFFSET default 1000)
 - Early stops fetching when requested date window is passed
 - Logs fetched timestamp range
 - Produces multiple PNGs (with safe try/except wrappers and placeholders):
     - shib_daily_volume_anomalies_annotated.png
     - shib_tx_size_histogram.png
     - shib_intertx_time_histogram.png
     - shib_wallet_hour_heatmap.png
     - shib_flow_graph_big_edges.png
 - Saves CSVs: shib_transfers_window.csv, shib_daily_aggregates.csv, shib_flagged_events.csv, shib_top_senders/receivers
 - Edit START_DATE_STR / END_DATE_STR / CONTRACT_ADDRESS as needed
"""

import os
import time
import math
import logging
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

# Logging + headless matplotlib backend (must be set before pyplot import)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.ensemble import IsolationForest

# ----------------- USER CONFIG -----------------
ETHERSCAN_API_KEY = "8BEVSDQF8511WEYYDAU8NKBS5SIZQ6Y4X2"
CONTRACT_ADDRESS = "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE"  # Shiba Inu (SHIB) token contract on Ethereum
START_DATE_STR = "15-08-2020"
END_DATE_STR   = "15-09-2020"
OUT_DIR = "shib_output_full"
# Pagination safety (Etherscan requires page*offset <= 10000)
PAGE_OFFSET = 1000   # safe default (set smaller if desired)
SORT_ORDER = 'asc'   # 'asc' -> old->new (allows early stop when last page ts > end_dt)
# thresholds and tuning
Z_THRESHOLD = 3.0
IF_CONTAMINATION = 0.02
TOP_WALLETS_HEATMAP = 25
EDGE_SHARE_THRESHOLD = 0.001
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def parse_ddmmyyyy(s):
    return datetime.strptime(s, "%d-%m-%Y")

start_dt = parse_ddmmyyyy(START_DATE_STR)
end_dt = parse_ddmmyyyy(END_DATE_STR) + timedelta(days=1) - timedelta(seconds=1)
logging.info(f"Window: {start_dt.isoformat()} -> {end_dt.isoformat()} (UTC)")

if ETHERSCAN_API_KEY == "YOUR_ETHERSCAN_API_KEY":
    logging.warning("ETHERSCAN_API_KEY not provided. Set environment var ETHERSCAN_API_KEY or edit script.")
    # We allow the script to run but Etherscan calls will fail; user should supply key.

# --------- Etherscan helpers (robust pagination) ----------
def fetch_token_transfers(api_key, contract_address, page=1, offset=1000, sort='asc'):
    base = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": contract_address,
        "page": page,
        "offset": offset,
        "sort": sort,
        "apikey": api_key
    }
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def collect_all_transfers(api_key, contract_address, start_dt=None, end_dt=None, sleep_between=0.2, sort='asc'):
    page = 1
    offset = PAGE_OFFSET
    rows = []
    while True:
        if page * offset > 10000:
            logging.info(f"Reached Etherscan page×offset safety limit (page={page}, offset={offset}). Stopping pagination.")
            break

        logging.info(f"Fetching page {page} (offset {offset}) ...")
        try:
            data = fetch_token_transfers(api_key, contract_address, page=page, offset=offset, sort=sort)
        except Exception as e:
            logging.exception("HTTP/requests error during fetch")
            break

        status = data.get("status")
        message = data.get("message","")
        results = data.get("result", [])

        if status == "0":
            logging.info(f"Etherscan returned status=0: {message}")
            # If it's a "Result window is too large" error and offset is big, reduce offset and retry once
            if "Result window is too large" in message and offset > 100:
                offset = min(500, max(100, offset // 2))
                logging.info(f"Reducing offset to {offset} and retrying page {page} ...")
                time.sleep(sleep_between)
                continue
            break

        if not results:
            logging.info("No results on this page; finishing fetch.")
            break

        # record timestamps to support early stopping
        try:
            ts_list = [int(r.get('timeStamp',0)) for r in results if r.get('timeStamp') is not None]
            if ts_list:
                page_min_ts = min(ts_list)
                page_max_ts = max(ts_list)
            else:
                page_min_ts = page_max_ts = None
        except Exception:
            page_min_ts = page_max_ts = None

        rows.extend(results)

        # Early stop heuristics:
        # If asc (old -> new) and last tx on this page is after end_dt, further pages will be beyond window.
        if end_dt is not None and sort == 'asc' and page_max_ts is not None:
            last_dt_page = datetime.utcfromtimestamp(page_max_ts)
            if last_dt_page > end_dt:
                logging.info("Last tx on page is after end_dt; stopping further pages.")
                break

        # If desc (new -> old) and first tx on this page is before start_dt, further pages are older.
        if start_dt is not None and sort == 'desc' and page_min_ts is not None:
            first_dt_page = datetime.utcfromtimestamp(page_min_ts)
            if first_dt_page < start_dt:
                logging.info("First tx on page is before start_dt; stopping further pages.")
                break

        if len(results) < offset:
            logging.info("Last page received (fewer than offset).")
            break

        page += 1
        time.sleep(sleep_between)

    logging.info(f"Collected {len(rows)} raw transfers.")
    df = pd.DataFrame(rows)
    if not df.empty and 'timeStamp' in df.columns:
        df['timeStamp'] = pd.to_datetime(df['timeStamp'].astype(int), unit='s', utc=True)
        logging.info("Fetched timestamps range: %s -> %s", df['timeStamp'].min(), df['timeStamp'].max())
    else:
        logging.info("No timestamp field present or dataframe empty.")
    return df

# ------------ Main pipeline ------------
logging.info("Starting fetch from Etherscan for SHIB...")
df = collect_all_transfers(ETHERSCAN_API_KEY, CONTRACT_ADDRESS, start_dt=start_dt, end_dt=end_dt, sort=SORT_ORDER)

if df.empty:
    logging.warning("No transfers fetched at all. Check contract address and API key.")
else:
    logging.info(f"Total transfers fetched: {len(df)}")

# Normalize and compute values
if not df.empty:
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], utc=True)
    df['value_raw'] = df['value'].astype(float)
    if 'tokenDecimal' in df.columns and df['tokenDecimal'].notnull().any():
        token_dec = int(df.loc[df['tokenDecimal'].notnull(), 'tokenDecimal'].iloc[0])
    else:
        token_dec = 18
    df['value'] = df['value_raw'] / (10 ** token_dec)
    df['date'] = df['timeStamp'].dt.floor('D')
    df['hour'] = df['timeStamp'].dt.hour
else:
    # create empty columns to avoid KeyErrors in later code
    df = pd.DataFrame(columns=['timeStamp','hash','from','to','value','value_raw','tokenDecimal','date','hour'])

# Focus window subset (strict inclusive)
mask_window = (df['timeStamp'] >= pd.Timestamp(start_dt, tz='UTC')) & (df['timeStamp'] <= pd.Timestamp(end_dt, tz='UTC'))
df_window = df.loc[mask_window].copy().reset_index(drop=True)
logging.info(f"SHIB transfers in requested window: {len(df_window)}")
df_window.to_csv(os.path.join(OUT_DIR, "shib_transfers_window.csv"), index=False)

# Daily aggregates across all fetched rows
if not df.empty:
    daily = df.groupby('date').agg(
        onchain_volume=('value','sum'),
        tx_count=('hash','nunique'),
        unique_senders=('from', lambda s: s.nunique()),
        unique_receivers=('to', lambda s: s.nunique())
    ).sort_index()
else:
    daily = pd.DataFrame(columns=['onchain_volume','tx_count','unique_senders','unique_receivers'])

# rolling z-score
daily['rolling_mean_30'] = daily['onchain_volume'].rolling(30, min_periods=7).mean()
daily['rolling_std_30']  = daily['onchain_volume'].rolling(30, min_periods=7).std().replace(0, np.nan)
daily['z_volume'] = (daily['onchain_volume'] - daily['rolling_mean_30']) / daily['rolling_std_30']
daily['z_flag'] = daily['z_volume'] > Z_THRESHOLD

# IsolationForest anomaly (if enough rows)
features = daily[['onchain_volume','tx_count','unique_senders','unique_receivers']].fillna(0)
if len(features) >= 20:
    if_model = IsolationForest(contamination=IF_CONTAMINATION, random_state=42)
    daily['if_pred'] = if_model.fit_predict(features)
    daily['if_anomaly'] = daily['if_pred'] == -1
else:
    daily['if_anomaly'] = False

daily.to_csv(os.path.join(OUT_DIR, "shib_daily_aggregates.csv"))

# Wallet-level and tx-level scoring inside the window
total_window_volume = df_window['value'].sum() if not df_window.empty else 0.0
large_tx_threshold = max(total_window_volume * 0.01, 1e-9)
logging.info(f"Total window volume: {total_window_volume:.6f} ; large_tx_threshold: {large_tx_threshold:.6f}")

if not df_window.empty:
    df_window = df_window.sort_values('timeStamp').reset_index(drop=True)
    df_window['prev_ts'] = df_window['timeStamp'].shift(1)
    df_window['inter_tx_seconds'] = (df_window['timeStamp'] - df_window['prev_ts']).dt.total_seconds().fillna(np.nan)

    wallet_stats = df_window.groupby('from').agg(
        total_sent=('value','sum'),
        tx_count=('hash','nunique'),
        first_seen=('timeStamp','min'),
        last_seen=('timeStamp','max')
    ).reset_index()
    wallet_stats['active_seconds'] = (wallet_stats['last_seen'] - wallet_stats['first_seen']).dt.total_seconds().replace(0,1)
    wallet_stats['txs_per_hour'] = wallet_stats['tx_count'] / (wallet_stats['active_seconds'] / 3600)

    def tx_suspicious_score(row, large_thresh, wallet_stats_df):
        score = 0.0
        reasons = []
        if row['value'] >= large_thresh:
            score += 0.5
            reasons.append("large_transfer")
        tiny_thresh = total_window_volume * 0.0005 if total_window_volume>0 else 0.001
        if row['value'] > 0 and row['value'] < tiny_thresh:
            score += 0.1
            reasons.append("tiny_transfer")
        if pd.notnull(row.get('inter_tx_seconds')) and row['inter_tx_seconds'] < 5:
            score += 0.15
            reasons.append("burst_timing")
        w = wallet_stats_df.loc[wallet_stats_df['from'] == row['from']]
        if not w.empty and w.iloc[0]['txs_per_hour'] > 50:
            score += 0.15
            reasons.append("sender_high_rate")
        return min(score, 1.0), ",".join(reasons)

    scored = []
    for _, r in df_window.iterrows():
        sc, rsn = tx_suspicious_score(r, large_tx_threshold, wallet_stats)
        scored.append((sc, rsn))
    df_window['suspicious_score'] = [s for s,_ in scored]
    df_window['suspicious_reasons'] = [r for _,r in scored]
else:
    df_window['suspicious_score'] = []
    df_window['suspicious_reasons'] = []

# export flagged events
flagged = df_window.loc[df_window['suspicious_score'] >= 0.5].copy()
flagged = flagged.sort_values(['suspicious_score','value'], ascending=[False, False])
flagged.to_csv(os.path.join(OUT_DIR, "shib_flagged_events.csv"), index=False)

# top senders / receivers
if not df_window.empty:
    senders = df_window.groupby('from').agg(sent_amount=('value','sum'), sent_count=('hash','nunique')).sort_values('sent_amount', ascending=False)
    receivers = df_window.groupby('to').agg(recv_amount=('value','sum'), recv_count=('hash','nunique')).sort_values('recv_amount', ascending=False)
    senders.head(100).to_csv(os.path.join(OUT_DIR, "shib_top_senders.csv"))
    receivers.head(100).to_csv(os.path.join(OUT_DIR, "shib_top_receivers.csv"))

# -------------- PLOTTING (safe wrappers) ----------------
def safe_save_figure(figfunc, out_path, *args, **kwargs):
    try:
        figfunc(*args, **kwargs)
        logging.info("Saved %s", out_path)
    except Exception as e:
        logging.exception("Failed to create %s", out_path)
        # attempt to create placeholder showing the error
        try:
            plt.figure(figsize=(6,3))
            plt.text(0.5, 0.5, f"Error creating image:\n{str(e)[:120]}", ha='center', va='center', fontsize=9)
            plt.axis('off')
            plt.savefig(out_path, dpi=100)
            plt.close()
        except Exception:
            pass

# daily time series annotated
def plot_daily_annotated():
    out_path = os.path.join(OUT_DIR, "shib_daily_volume_anomalies_annotated.png")
    try:
        plt.figure(figsize=(16,8))
        if not daily.empty:
            # Prepare data
            dates = daily.index
            vols = daily['onchain_volume'].fillna(0)
            tx_counts = daily['tx_count'].fillna(0)

            # Plot volume as bars
            ax = plt.gca()
            ax.bar(dates, vols, width=0.8, alpha=0.85, label='On-chain Volume', edgecolor='none', color='#4A90E2')

            # Overlay transaction count as a thin gray line on secondary axis
            ax2 = ax.twinx()
            scale = vols.max() / (tx_counts.max() + 1) if tx_counts.max() > 0 else 1
            ax2.plot(dates, tx_counts * scale, color='gray', linewidth=1.0, alpha=0.6, label='Tx Count (scaled)')
            ax2.set_ylabel("Tx Count (scaled)")

            # Grid and labels
            ax.set_title("SHIB — Daily On-Chain Volume with Detected Anomalies", fontsize=16)
            ax.set_ylabel("Token Volume")
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            # Highlight requested window
            ax.axvspan(start_dt, end_dt, color='#FFECB3', alpha=0.25, label='Requested Window')

            # Mark anomalies with red dots and boxed annotations
            for dt, row in daily.iterrows():
                y = row['onchain_volume']
                if row.get('z_flag', False) or row.get('if_anomaly', False):
                    ax.plot(dt, y, marker='o', markersize=9, color='red', zorder=6)
                    reasons = []
                    if row.get('z_flag', False): 
                        reasons.append(f"z>{Z_THRESHOLD}")
                    if row.get('if_anomaly', False): 
                        reasons.append("IsolationForest")
                    
                    txt = " & ".join(reasons)
                    ax.annotate("Suspicious: " + txt,
                                xy=(dt, y),
                                xytext=(0, 20),
                                textcoords='offset points',
                                ha='center', va='bottom',
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9),
                                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

            # Format the X-axis nicely
            ax.xaxis.set_tick_params(rotation=30)
            ax.xaxis.set_major_locator(plt.MaxNLocator(12))

            # Combine legends from both axes
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=9)

            # Adjust limits for clarity
            ymin = 0
            ymax = max(vols.max() * 1.4, 1)
            ax.set_ylim(ymin, ymax)

        else:
            plt.text(0.5, 0.5, "No SHIB daily on-chain data available", ha='center', va='center', fontsize=12)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    except Exception:
        logging.exception("plot_daily_annotated failed")
        plt.figure(figsize=(8,3))
        plt.text(0.5, 0.5, "Error creating daily annotated plot", ha='center', va='center')
        plt.axis('off')
        plt.savefig(out_path, dpi=150)
        plt.close()

safe_save_figure(plot_daily_annotated, os.path.join(OUT_DIR, "shib_daily_volume_anomalies_annotated.png"))

# inter-tx time histogram
def plot_intertx_hist():
    out_path = os.path.join(OUT_DIR, "shib_intertx_time_histogram.png")
    if not df_window.empty and 'inter_tx_seconds' in df_window.columns:
        plt.figure(figsize=(8,4))
        its = df_window['inter_tx_seconds'].dropna()
        if its.empty:
            plt.text(0.5, 0.5, "No inter-tx intervals available", ha='center', va='center')
            plt.axis('off')
            plt.savefig(out_path, dpi=150)
            plt.close()
            return
        plt.hist(its.clip(upper=3600), bins=80, log=True)
        plt.title("Histogram of inter-transaction times (seconds) - capped to 3600s")
        plt.xlabel("Seconds between consecutive transfers")
        plt.ylabel("Count (log)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.figure(figsize=(6,3))
        plt.text(0.5, 0.5, "No SHIB transfer data in window\n(inter-tx histogram)", ha='center', va='center')
        plt.axis('off')
        plt.savefig(out_path, dpi=150)
        plt.close()

safe_save_figure(plot_intertx_hist, os.path.join(OUT_DIR, "shib_intertx_time_histogram.png"))

# wallet-hour heatmap
def plot_wallet_heatmap():
    out_path = os.path.join(OUT_DIR, "shib_wallet_hour_heatmap.png")
    if not df_window.empty:
        sent = df_window.groupby('from')['value'].sum()
        recv = df_window.groupby('to')['value'].sum()
        total_activity = sent.add(recv, fill_value=0).sort_values(ascending=False)
        top_wallets = total_activity.head(TOP_WALLETS_HEATMAP).index.tolist()
        if len(top_wallets) == 0:
            raise ValueError("No top wallets for heatmap")
        df_heat = df_window.loc[(df_window['from'].isin(top_wallets)) | (df_window['to'].isin(top_wallets))].copy()
        rows = []
        for w in top_wallets:
            mask = (df_heat['from'] == w) | (df_heat['to'] == w)
            sub = df_heat.loc[mask]
            counts = sub.groupby('hour')['hash'].nunique()
            row = [int(counts.get(h,0)) for h in range(24)]
            rows.append(row)
        heat_df = pd.DataFrame(rows, index=[w[:12] for w in top_wallets], columns=list(range(24)))
        plt.figure(figsize=(12, max(4, len(top_wallets)*0.25 + 2)))
        sns.heatmap(heat_df, annot=False, fmt="d", cbar=True)
        plt.title("SHIB wallet activity heatmap (top wallets) — tx counts by hour (UTC)")
        plt.xlabel("Hour of day (UTC)")
        plt.ylabel("Wallet (short)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        plt.figure(figsize=(6,3))
        plt.text(0.5, 0.5, "No SHIB transfers in window\n(wallet heatmap)", ha='center', va='center')
        plt.axis('off')
        plt.savefig(out_path, dpi=150)
        plt.close()

safe_save_figure(plot_wallet_heatmap, os.path.join(OUT_DIR, "shib_wallet_hour_heatmap.png"))

def plot_flow_graph():
    out_path = os.path.join(OUT_DIR, "shib_flow_graph_big_edges.png")
    try:
        plt.figure(figsize=(14,10))
        if not df_window.empty:
            edge_agg = df_window.groupby(['from','to']).agg(amount=('value','sum')).reset_index()
            big_edges = edge_agg[edge_agg['amount'] >= (total_window_volume * EDGE_SHARE_THRESHOLD)]
            if big_edges.empty:
                plt.text(0.5, 0.5, "No large transfer edges above threshold", ha='center', va='center', fontsize=12)
                plt.axis('off')
                plt.savefig(out_path, dpi=200)
                plt.close()
                return

            # build graph
            G = nx.DiGraph()
            for _, r in big_edges.iterrows():
                G.add_edge(r['from'], r['to'], weight=float(r['amount']))

            # compute node flow sizes (sum of in+out weights)
            node_weights = {}
            for n in G.nodes():
                in_w = sum(d.get('weight',0) for _,_,d in G.in_edges(n, data=True))
                out_w = sum(d.get('weight',0) for _,_,d in G.out_edges(n, data=True))
                node_weights[n] = in_w + out_w

            # scale node sizes for plotting
            max_node_w = max(node_weights.values()) if node_weights else 1
            node_size = {n: 300 + 2000 * (w / max_node_w) for n,w in node_weights.items()}

            # choose top nodes to label (to avoid label clutter)
            top_nodes_by_weight = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)
            label_nodes = set([n for n,_ in top_nodes_by_weight[:20]])  # label top 20

            # layout - spring layout with scale tuned to node count
            k = 0.3 if len(G) < 50 else 0.1
            pos = nx.spring_layout(G, k=k, seed=42, iterations=200)

            # draw nodes
            sizes = [node_size.get(n, 300) for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=sizes, alpha=0.9)
            # draw edges with widths scaled by weight, and partial transparency
            weights = [d['weight'] for _,_,d in G.edges(data=True)]
            max_w = max(weights) if weights else 1.0
            widths = [1.0 + 6.0 * (w / max_w) for w in weights]
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=widths, alpha=0.6, arrowsize=12, arrowstyle='-|>')

            # labels only for top nodes
            labels = {n: (n[:12] + "...") for n in G.nodes() if n in label_nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

            plt.title("SHIB flow graph (window) — large edges (node size ∝ flow)", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
        else:
            plt.text(0.5, 0.5, "No SHIB transfers in window\n(flow graph)", ha='center', va='center')
            plt.axis('off')
            plt.savefig(out_path, dpi=200)
            plt.close()
    except Exception:
        logging.exception("plot_flow_graph failed")
        plt.figure(figsize=(8,3))
        plt.text(0.5,0.5,"Error creating flow graph", ha='center', va='center')
        plt.axis('off')
        plt.savefig(out_path, dpi=150)
        plt.close()

safe_save_figure(plot_flow_graph, os.path.join(OUT_DIR, "shib_flow_graph_big_edges.png"))

logging.info("All SHIB visuals and CSVs saved to %s", OUT_DIR)
logging.info("Summary: shib_transfers_in_window=%d ; shib_flagged_events=%d", len(df_window), len(flagged))
logging.info("Top SHIB flagged events (first 10):")
if not flagged.empty:
    logging.info(flagged[['timeStamp','hash','from','to','value','suspicious_score','suspicious_reasons']].head(10).to_string(index=False))
else:
    logging.info("No SHIB flagged events (score >= 0.5) found in the window.")
