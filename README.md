# KXNBAGAME-26FEB28TORWAS — Toronto Raptors at Washington Wizards (Feb 28, 2026)

## Event Overview
- **Sport**: NBA Basketball (Pro Basketball M)
- **Type**: Moneyline (who wins the game)
- **Markets**: 2 binary contracts
  - `KXNBAGAME-26FEB28TORWAS-WAS` — "Will Washington win?"
  - `KXNBAGAME-26FEB28TORWAS-TOR` — "Will Toronto win?"
- **Note**: These are mutually exclusive binary contracts. YES price = probability.
  WAS YES at 65c means market implies 65% chance Washington wins.

## Data Collection Period
- **Start**: 2026-02-26 (market discovered)
- **End**: 2026-03-01 (post-settlement)
- **Source**: Kalshi Exchange real-time WebSocket feed

## Data Volume Summary
- **Trades**: ~40,679 unique executions (WAS: 29,506 / TOR: 11,173)
- **Deltas**: ~306,967 order book changes (WAS: 171,824 / TOR: 135,143)
- **Snapshots**: Periodic L2 snapshots (~every 30 seconds)

## File Descriptions

### `event_metadata.csv`
Event-level info: title, category, mutually exclusive flag, series ticker.

### `market_metadata.csv`
Per-market details: title, status, result, volume, open interest, last price,
open/close times, resolution rules.

### `trades.csv`
Every trade execution on both markets.
| Column | Description |
|--------|-------------|
| datetime_utc | Human-readable timestamp (UTC) |
| ts | Unix timestamp (seconds) |
| trade_id | Unique trade identifier |
| market_ticker | Which market (WAS or TOR) |
| yes_price | YES price in cents (1-99) |
| no_price | NO price in cents (100 - yes_price) |
| count | Number of contracts traded |
| taker_side | "yes" or "no" — which side the taker bought |

### `deltas/deltas_WAS.csv`, `deltas/deltas_TOR.csv`
Every single order book change (tick-by-tick).
| Column | Description |
|--------|-------------|
| time | Timestamp of the change (UTC) |
| side | "yes" or "no" — which side of the book changed |
| price | Price level that changed (in dollars, e.g., 0.65) |
| size | New size at that level (0 = level removed) |
| seq | Sequence number — apply in order to reconstruct exact book state |
| action | "update" (new/changed level) or "remove" (level gone) |

**To reconstruct the order book at any point in time:**
1. Start from the nearest snapshot before your target time
2. Apply all deltas (in seq order) up to your target time
3. Result = exact order book state at that moment

### `orderbook_snapshots/snapshots_WAS.csv`, `snapshots_TOR.csv`
Periodic L2 order book snapshots (~every 30 seconds).
Includes full bid/ask arrays (JSON), top-of-book prices, spread, imbalance.

### `reconstructed_orderbook/reconstructed_WAS.csv`, `reconstructed_TOR.csv`
Time series of top-of-book metrics at each snapshot:
| Column | Description |
|--------|-------------|
| timestamp | Snapshot time (UTC) |
| best_bid / best_ask | Top-of-book prices (dollars) |
| mid_price | (best_bid + best_ask) / 2 |
| micro_price | Size-weighted mid price |
| spread | best_ask - best_bid (dollars) |
| imbalance | (bid_vol - ask_vol) / total_vol |
| best_bid_size / best_ask_size | Contracts at top level |
| bid_count / ask_count | Number of price levels |
| *_cents | Same prices in cents for convenience |

### `reconstructed_orderbook/depth_WAS.csv`, `depth_TOR.csv`
Full order book depth at each snapshot — one row per price level:
| Column | Description |
|--------|-------------|
| timestamp | Snapshot time |
| side | "bid" or "ask" |
| level | Depth level (0 = best) |
| price_cents | Price in cents (1-99) |
| price_dollars | Price in dollars (0.01-0.99) |
| size | Number of contracts at this level |

## Kalshi Binary Market Mechanics
- Each contract pays $1 if YES, $0 if NO
- YES price = implied probability (65c = 65% chance)
- YES + NO prices ~ $1.00 (with spread)
- Moneyline: "Will [team] win?" — YES = team wins, NO = team loses
- Mutually exclusive: WAS YES ~ TOR NO (but can differ due to spread/timing)
