"""Constant-product AMM math for Bittensor subnet alpha pools.

The invariant is k = tao_in * alpha_in. There are no swap fees. All functions
here are pure and take pool state as arguments.
"""

from __future__ import annotations


def spot_price(tao_in: float, alpha_in: float) -> float:
    """tao_in / alpha_in, or 0 if the pool is empty."""
    if alpha_in <= 0 or tao_in <= 0:
        return 0.0
    return tao_in / alpha_in


def buy_alpha(dx_tao: float, tao_in: float, alpha_in: float) -> dict:
    """Stake dx_tao TAO, receive alpha tokens.

    Returns alpha_received, spot_price, effective_price, slippage_pct,
    and the new pool reserves.
    """
    if dx_tao <= 0 or tao_in <= 0 or alpha_in <= 0:
        return {
            "alpha_received": 0.0,
            "spot_price": spot_price(tao_in, alpha_in),
            "effective_price": 0.0,
            "slippage_pct": 0.0,
            "new_tao_in": tao_in,
            "new_alpha_in": alpha_in,
        }
    k = tao_in * alpha_in
    new_tao_in = tao_in + dx_tao
    new_alpha_in = k / new_tao_in
    alpha_received = alpha_in - new_alpha_in
    sp = tao_in / alpha_in
    ep = dx_tao / alpha_received if alpha_received > 0 else 0.0
    slip = (ep / sp) - 1.0 if sp > 0 and ep > 0 else 0.0
    return {
        "alpha_received": alpha_received,
        "spot_price": sp,
        "effective_price": ep,
        "slippage_pct": slip,
        "new_tao_in": new_tao_in,
        "new_alpha_in": new_alpha_in,
    }


def sell_alpha(da_alpha: float, tao_in: float, alpha_in: float) -> dict:
    """Unstake da_alpha alpha, receive TAO."""
    if da_alpha <= 0 or tao_in <= 0 or alpha_in <= 0:
        return {
            "tao_received": 0.0,
            "spot_price": spot_price(tao_in, alpha_in),
            "effective_price": 0.0,
            "slippage_pct": 0.0,
            "new_tao_in": tao_in,
            "new_alpha_in": alpha_in,
        }
    k = tao_in * alpha_in
    new_alpha_in = alpha_in + da_alpha
    new_tao_in = k / new_alpha_in
    tao_received = tao_in - new_tao_in
    sp = tao_in / alpha_in
    ep = tao_received / da_alpha if da_alpha > 0 else 0.0
    slip = 1.0 - (ep / sp) if sp > 0 and ep > 0 else 0.0
    return {
        "tao_received": tao_received,
        "spot_price": sp,
        "effective_price": ep,
        "slippage_pct": slip,
        "new_tao_in": new_tao_in,
        "new_alpha_in": new_alpha_in,
    }


def max_tao_for_slippage(max_slippage: float, tao_in: float, alpha_in: float) -> float:
    """Max TAO to stake while keeping buy-side slippage below max_slippage.

    For x*y=k with no fees, buying dx TAO yields alpha_out = alpha_in*dx/(tao_in+dx).
    Effective price is dx/alpha_out = (tao_in+dx)/alpha_in.
    Slippage = effective/spot - 1 = dx/tao_in.
    So dx_max = max_slippage * tao_in.
    """
    if max_slippage <= 0 or tao_in <= 0 or alpha_in <= 0:
        return 0.0
    return max_slippage * tao_in


def simulate_roundtrip(
    tao_in_entry: float,
    alpha_in_entry: float,
    tao_in_exit: float,
    alpha_in_exit: float,
    tao_amount: float,
) -> dict:
    """Simulate buying at entry pool state, then selling at exit pool state.

    Entry and exit pool states are independent, other participants move the
    pool between the two events.
    """
    buy = buy_alpha(tao_amount, tao_in_entry, alpha_in_entry)
    alpha_bought = buy["alpha_received"]
    entry_slippage_pct = buy["slippage_pct"]

    sell = sell_alpha(alpha_bought, tao_in_exit, alpha_in_exit)
    tao_received = sell["tao_received"]
    exit_slippage_pct = sell["slippage_pct"]

    net_pnl_tao = tao_received - tao_amount
    net_return_pct = net_pnl_tao / tao_amount if tao_amount > 0 else 0.0

    # Slippage cost: entry pays (dx*slip), exit loses (tao_received_frictionless - tao_received)
    entry_slip_cost = tao_amount * entry_slippage_pct
    exit_slip_cost = (tao_received / (1 - exit_slippage_pct) - tao_received) if exit_slippage_pct < 1 else 0.0
    total_slippage_cost_tao = entry_slip_cost + exit_slip_cost

    return {
        "alpha_bought": alpha_bought,
        "entry_slippage_pct": entry_slippage_pct,
        "tao_received": tao_received,
        "exit_slippage_pct": exit_slippage_pct,
        "net_pnl_tao": net_pnl_tao,
        "net_return_pct": net_return_pct,
        "total_slippage_cost_tao": total_slippage_cost_tao,
    }
