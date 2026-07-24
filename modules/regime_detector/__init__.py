"""
Regime Detector module — turns enriched candle parquets into regime label
files: one parquet per trading day under
{data_root}/regimes/{ASSET}/{run_name}/YYYY-MM-DD.parquet, one row per
snapshot, each row built ONLY from bars strictly before its timestamp.

CONSUMER WARNING (read before wiring the Backtester / Analytics / Monte
Carlo to these files): the FINAL row of each file knows how the day ended.
Joining trades to regimes on date alone gives every trade the hindsight row
and turns the backtest into fiction. The correct join is

    pd.merge_asof(trades, regimes, left_on=<entry timestamp>,
                  right_index=True, direction="backward")

so each trade sees the latest snapshot at or before its entry. Build that
helper ONCE in modules/common/backend/ and share it — do not hand-roll the
merge per consumer.
"""
