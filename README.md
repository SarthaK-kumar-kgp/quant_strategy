# Quant Strategy

This repository contains two independent Polymarket quantitative-strategy
research projects. Each folder has a different signal source and should be
treated as a separate strategy.

## Repository structure

### [`whaling_strategy/`](whaling_strategy/)

A wallet-following research project that studies whether historically strong
or unusually informed Polymarket wallets can provide useful trading signals.
It includes:

- wallet and market data collectors;
- wallet scoring and registry logic;
- smart-money and insider-activity strategies;
- historical backtesting and analysis scripts; and
- supporting strategy documentation.

Start with the [whaling strategy README](whaling_strategy/README.md).

### [`cross-market-strategy/`](cross-market-strategy/)

A crypto barrier strategy that estimates the probability of BTC, ETH, SOL, or
XRP reaching a specified price barrier and compares that estimate with the
corresponding Polymarket price. Its code is organized into sequential research
phases:

1. build and audit the eligible contract registry;
2. construct causal datasets and walk-forward folds;
3. fit, calibrate, and compare probability models;
4. evaluate execution-stressed trades; and
5. simulate portfolio capital, exposure, and risk controls.

This repository intentionally contains only the Python source and test files
for this strategy. Historical datasets, generated predictions, model outputs,
reports, logs, and portfolio results are not included.

## Important note

These projects are research prototypes. Historical probability scores and
backtests do not establish that quoted prices were executable or that a
strategy will remain profitable after live fees, latency, slippage, and
available order-book depth.
