"""Store–region–network hierarchical forecasting layer.

This package is additive: single-series upload/fit/backtest behaviour lives
unchanged in the surrounding modules.  The hierarchy layer only organises
existing leaf series into a tree (max depth 3), aggregates their history, and
reconciles their base forecasts (see kernels/reconcile.py for the chosen
method).
"""
