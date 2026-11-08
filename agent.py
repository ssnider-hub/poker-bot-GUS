"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "404 Brain Not Found"
AGENT_NAME = "Gus Hansen"


# ===== YOUR CODE HERE =====

import time
import random
from evaluator import (
    card_rank, card_suit, omaha_score, get_table,
    _PAIRS, _TRIPLES
)
from game import Action, AgentConfig, HandResult, Observation, Street

# Target thinking time: 6.2s out of 8.0s (leaving a robust ~1.8s / 22.5% cushion)
TARGET_TOTAL_TIME = 6.2
SAFETY_FLOOR_SECONDS = 1.65


def new_half(config: AgentConfig, memory: dict) -> None:
    """Reset match metrics and time trackers at the start of each half."""
    memory["hands_played"] = 0
    memory["time_charged"] = 0.0
    memory["decisions"] = 0
    memory["opp_folds"] = 0


def _evaluate_preflop_fast(hole_cards: tuple[int, ...]) -> float:
    """Enhanced preflop evaluator (< 0.04 ms).
    
    Evaluates:
    - High broadway density
    - High pairs (AA/KK weighted heavily)
    - Double-suitedness (2+2 is premium; 4 of same suit penalized because it blocks outs)
    - Wraps / Rundowns (5-card straights in Short Deck)
    """
    ranks = sorted([c >> 2 for c in hole_cards], reverse=True)
    suits = [c & 3 for c in hole_cards]
    
    # 1. Base high-card power
    score = sum(ranks) / 36.0
    
    # 2. Pairs, sets, quads
    r_counts = {}
    for r in ranks:
        r_counts[r] = r_counts.get(r, 0) + 1
    for r, cnt in r_counts.items():
        if cnt == 2:
            score += 0.22 + (r / 8.0) * 0.18  # AA gives +0.40, KK +0.37
        elif cnt >= 3:
            score += 0.38

    # 3. Suitedness (Double-suited is the gold standard)
    s_counts = {}
    for s in suits:
        s_counts[s] = s_counts.get(s, 0) + 1
    
    double_suited = sum(1 for cnt in s_counts.values() if cnt == 2) >= 2
    if double_suited:
        score += 0.25
    else:
        score += sum(0.12 for cnt in s_counts.values() if cnt in (2, 3))
    
    # Holding 4 or 5 of the same suit is bad: you block your own outs
    if any(cnt >= 4 for cnt in s_counts.values()):
        score -= 0.15

    # 4. Connectivity / Rundowns (e.g., 7-8-9-T-J or 8-9-T-J-Q)
    unique_ranks = sorted(list(set(ranks)))
    if len(unique_ranks) >= 4:
        span = unique_ranks[-1] - unique_ranks[0]
        if span <= 5:
            score += 0.20

    return min(1.0, max(0.05, score))


def choose_action(obs: Observation, memory: dict) -> Action:
    """Dynamically scales rollouts to push near 8s while guaranteeing safety."""
    t_start = time.perf_counter()
    action = Action.CHECK_CALL
    try:
        # --- 1. Dynamic Time Bank Management ---
        hands_left = max(1, 1000 - obs.hand_index)
        time_charged = memory.get("time_charged", 0.0)
        bank_left = max(0.0, 8.0 - time_charged)

        if bank_left < SAFETY_FLOOR_SECONDS:
            # Emergency Cushion Mode: < 0.95ms is 100% free under the 1ms network grace
            max_dt = 0.00095
            max_rollouts = 15
        else:
            # Budget dynamically toward the 6.2s target
            budget_left = max(0.1, TARGET_TOTAL_TIME - time_charged)
            avg_decisions_left = max(10, hands_left * 2.3)
            base_dt = budget_left / avg_decisions_left

            # Allocate deeper thinking to late streets (Turn/River) and big pots
            pot_weight = 1.35 if obs.pot > 16 else 0.8
            street_weight = 1.45 if obs.street in (Street.TURN, Street.RIVER) else 0.85
            max_dt = min(0.015, max(0.0011, (base_dt * pot_weight * street_weight) + 0.00095))
            max_rollouts = int(max_dt * 30000)

        deadline = t_start + max_dt
        table = get_table()

        # --- 2. Pot Odds & Opponent Profile ---
        to_call = obs.to_call
        pot_after_call = obs.pot + to_call
        pot_odds = (to_call / float(pot_after_call)) if pot_after_call > 0 else 0.0

        hands_seen = memory.get("hands_played", 0)
        opp_folds = memory.get("opp_folds", 0)
        opp_fold_freq = (opp_folds / float(hands_seen)) if hands_seen > 15 else 0.25
        is_passive_opp = opp_fold_freq > 0.38

        # --- 3. High-Speed Equity Engine ---
        if obs.street == Street.PREFLOP:
            equity = _evaluate_preflop_fast(obs.hole_cards)

        elif obs.street == Street.RIVER:
            # River: All 5 board cards are static!
            my_sc = omaha_score(obs.hole_cards, obs.board)
            bbits = [1 << c for c in obs.board]
            bmasks = [bbits[i] | bbits[j] | bbits[k] for i, j, k in _TRIPLES]
            known = set(obs.hole_cards) | set(obs.board)
            deck = [c for c in range(36) if c not in known]

            wins, total = 0, 0
            while time.perf_counter() < deadline and total < max(40, max_rollouts):
                opp_hole = random.sample(deck, 5)
                # Opponent Range Conditioning: If opp raised/bet, weight towards stronger hands
                if obs.opp_bet > obs.big_blind and random.random() < 0.50:
                    if max(c >> 2 for c in opp_hole) < 6 and len(set(c >> 2 for c in opp_hole)) == 5:
                        continue
                
                obits = [1 << c for c in opp_hole]
                omasks = [obits[i] | obits[j] for i, j in _PAIRS]
                opp_sc = max(table[om | bm] for om in omasks for bm in bmasks)
                
                if my_sc > opp_sc:
                    wins += 1
                elif my_sc == opp_sc:
                    wins += 0.5
                total += 1
            equity = (wins / float(total)) if total > 0 else 0.5

        else:
            # Flop / Turn: Precompute own hole masks once
            known = set(obs.hole_cards) | set(obs.board)
            deck = [c for c in range(36) if c not in known]
            needed_board = 5 - len(obs.board)

            my_bits = [1 << c for c in obs.hole_cards]
            my_masks = [my_bits[i] | my_bits[j] for i, j in _PAIRS]

            wins, total = 0, 0
            while time.perf_counter() < deadline and total < max(30, max_rollouts):
                sample = random.sample(deck, needed_board + 5)
                sim_b = list(obs.board) + sample[:needed_board]
                opp_hole = sample[needed_board:]

                bbits = [1 << c for c in sim_b]
                bmasks = [bbits[i] | bbits[j] | bbits[k] for i, j, k in _TRIPLES]
                my_sc = max(table[hm | bm] for hm in my_masks for bm in bmasks)

                obits = [1 << c for c in opp_hole]
                omasks = [obits[i] | obits[j] for i, j in _PAIRS]
                opp_sc = max(table[om | bm] for om in omasks for bm in bmasks)

                if my_sc > opp_sc:
                    wins += 1
                elif my_sc == opp_sc:
                    wins += 0.5
                total += 1
            equity = (wins / float(total)) if total > 0 else 0.5

        # --- 4. Positional Advantage & Decision Matrix ---
        in_position = obs.is_button if obs.street != Street.PREFLOP else False
        pos_boost = 0.04 if in_position else 0.0

        # Monster Hand: Max pot-limit raise
        if (equity + pos_boost) >= 0.71 and Action.RAISE_POT in obs.legal_actions:
            action = Action.RAISE_POT
        # Strong Hand: Half-pot value raise
        elif (equity + pos_boost) >= 0.53 and Action.RAISE_HALF in obs.legal_actions and obs.raises_this_street < 2:
            action = Action.RAISE_HALF
        # Exploitative Bluff: Probe bet against folding bots when checked to on Turn/River
        elif is_passive_opp and to_call == 0 and obs.street in (Street.TURN, Street.RIVER) and random.random() < 0.20:
            action = Action.RAISE_HALF if Action.RAISE_HALF in obs.legal_actions else Action.CHECK_CALL
        # Positive Expected Value (EV): Call when equity beats pot odds
        elif equity >= (pot_odds - 0.04):
            action = Action.CHECK_CALL
        # Fold negative-EV bets, check if free
        elif Action.FOLD in obs.legal_actions:
            action = Action.FOLD
        else:
            action = Action.CHECK_CALL

    except Exception:
        # Emergency Safe Net: Never crash or forfeit
        if obs.to_call == 0:
            action = Action.CHECK_CALL
        elif Action.FOLD in obs.legal_actions:
            action = Action.FOLD
        else:
            action = Action.CHECK_CALL

    finally:
        # Time accounting: The first 1ms of every request is free (network grace)
        duration = time.perf_counter() - t_start
        charged = max(0.0, duration - 0.001)
        memory["time_charged"] = memory.get("time_charged", 0.0) + charged
        memory["decisions"] = memory.get("decisions", 0) + 1

    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Learn from previous hands across the 1,000-hand half."""
    try:
        memory["hands_played"] = memory.get("hands_played", 0) + 1
        # Track whether opponent folds to our bets
        if not result.went_to_showdown and result.winner == "me":
            memory["opp_folds"] = memory.get("opp_folds", 0) + 1
    except Exception:
        pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
