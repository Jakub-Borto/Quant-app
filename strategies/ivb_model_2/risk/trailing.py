"""
Trailing stop logic — placeholder for future development.

Roadmap ideas (from documentation):
  - Trail SL to absorption level when in profit
  - Minimum risk threshold — extend TP when SL too close

Intended interface (to be wired into run_trade later):

    def trail_stop(post_entry, entry_ts, entry_price, direction, sl, tp, params) -> float:
        '''Return an updated SL given current open trade context.'''
        ...
"""
