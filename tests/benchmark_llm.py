#!/usr/bin/env python3
"""
benchmark_llm.py — LLM benchmark harness for Air Strip One.

Runs Mistral-small-latest (or any compatible model) as both Oceania and
Eurasia commanders in a turn-based strategic war.

Usage:
    export MISTRAL_API_KEY="your-key-here"
    python tests/benchmark_llm.py --model mistral-small-latest --turns 50
    python tests/benchmark_llm.py --model mistral-small-latest --turns 30 --oceania-only
    python tests/benchmark_llm.py --dry-run --turns 10  # no API calls, random actions

Token budget per turn:
  - System prompt: ~300 tokens (sent once)
  - Turn summary: ~500-800 tokens
  - LLM response: ~100-200 tokens
  - Total per turn per faction: ~700-1200 tokens
  - 50 turns × 2 factions = ~70K-120K tokens total (well within limits)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Load .env file ────────────────────────────────────────────────────────── #
def _load_env(env_path: str = None):
    """Load .env file into os.environ. No external dependency needed."""
    path = Path(env_path) if env_path else _ROOT / ".env"
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:  # don't override existing env vars
                os.environ[key] = value

_load_env()

import numpy as np

from gravitas.llm_game import (
    GameState, create_game, step_game,
    summarize_turn, parse_action, apply_actions,
    SYSTEM_PROMPT, COMMENTARY_SYSTEM_PROMPT,
    OCEANIA_SYSTEM_PROMPT, EURASIA_SYSTEM_PROMPT,
    WINSTON_SYSTEM_PROMPT,
    summarize_blf_turn, parse_blf_action, apply_blf_actions,
    generate_visible_events, format_commentary_prompt,
)


# ═══════════════════════════════════════════════════════════════════════════ #
# LLM Client                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class LLMClient:
    """Multi-provider LLM client: Mistral, Anthropic, or dry-run."""

    def __init__(self, model: str = "mistral-small-latest", dry_run: bool = False,
                 max_tokens: int = 1024, label: str = ""):
        self.model = model
        self.dry_run = dry_run
        self.max_tokens = max_tokens
        self.label = label or model
        self.client = None
        self.provider = "none"  # "mistral", "anthropic", "none"
        # Token tracking
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        # Call tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retried_calls = 0
        # Latency tracking
        self.latencies: List[float] = []
        # Action tracking
        self.actions_parsed = 0
        self.actions_noop = 0

        if not dry_run:
            if "claude" in model:
                self._init_anthropic()
            else:
                self._init_mistral()

    def _init_mistral(self):
        try:
            from mistralai.client import Mistral
            api_key = os.environ.get("MISTRAL_API_KEY", "")
            if not api_key:
                print(f"WARNING: MISTRAL_API_KEY not set for model {self.model}. Dry-run.")
                self.dry_run = True
            else:
                self.client = Mistral(api_key=api_key)
                self.provider = "mistral"
        except ImportError:
            print("WARNING: mistralai not installed. pip install mistralai")
            self.dry_run = True

    def _init_anthropic(self):
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                print(f"WARNING: ANTHROPIC_API_KEY not set for model {self.model}. Dry-run.")
                self.dry_run = True
            else:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.provider = "anthropic"
        except ImportError:
            print("WARNING: anthropic not installed. pip install anthropic")
            self.dry_run = True

    def chat(self, system: str, user: str, max_retries: int = 5) -> str:
        """Send a chat message with automatic retry on failure. 5s sleep between retries."""
        self.total_calls += 1

        if self.dry_run:
            return self._random_action()

        t_start = time.time()
        for attempt in range(max_retries):
            try:
                if self.provider == "mistral":
                    result = self._chat_mistral(system, user)
                elif self.provider == "anthropic":
                    result = self._chat_anthropic(system, user)
                else:
                    result = "NOOP"
                latency = time.time() - t_start
                self.latencies.append(latency)
                self.successful_calls += 1
                return result
            except Exception as e:
                if attempt > 0:
                    self.retried_calls += 1
                is_last = attempt == max_retries - 1
                err_short = str(e)[:120]
                if is_last:
                    self.failed_calls += 1
                    self.latencies.append(time.time() - t_start)
                    print(f"  API FAILED after {max_retries} retries ({self.provider}): {err_short}")
                    return "NOOP"
                else:
                    wait = 5 * (attempt + 1)
                    print(f"  API retry {attempt+1}/{max_retries} ({self.provider}): {err_short} — waiting {wait}s...")
                    time.sleep(wait)

    def stats(self) -> Dict:
        """Return comprehensive statistics for this client."""
        lats = self.latencies if self.latencies else [0.0]
        return {
            "model": self.model,
            "provider": self.provider,
            "label": self.label,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "retried_calls": self.retried_calls,
            "success_rate": f"{self.successful_calls / max(self.total_calls, 1) * 100:.1f}%",
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out,
            "tokens_total": self.total_tokens_in + self.total_tokens_out,
            "avg_tokens_per_call_in": round(self.total_tokens_in / max(self.successful_calls, 1)),
            "avg_tokens_per_call_out": round(self.total_tokens_out / max(self.successful_calls, 1)),
            "latency_avg_s": round(sum(lats) / len(lats), 2),
            "latency_min_s": round(min(lats), 2),
            "latency_max_s": round(max(lats), 2),
            "latency_p50_s": round(sorted(lats)[len(lats) // 2], 2),
            "latency_p95_s": round(sorted(lats)[int(len(lats) * 0.95)], 2) if len(lats) > 1 else round(lats[0], 2),
            "actions_parsed": self.actions_parsed,
            "actions_noop": self.actions_noop,
        }

    def _chat_mistral(self, system: str, user: str) -> str:
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self.max_tokens,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        if hasattr(response, "usage") and response.usage:
            self.total_tokens_in += response.usage.prompt_tokens
            self.total_tokens_out += response.usage.completion_tokens
        return content

    def _chat_anthropic(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self.max_tokens,
            temperature=0.7,
        )
        content = response.content[0].text
        if hasattr(response, "usage") and response.usage:
            self.total_tokens_in += response.usage.input_tokens
            self.total_tokens_out += response.usage.output_tokens
        return content

    def _random_action(self) -> str:
        """Generate random plausible actions for dry-run mode."""
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        actions = [
            "SET_MANUFACTURING 0.6", "BUILD_SHIP DESTROYER",
            "BUILD_SQUADRON INTERCEPTOR", "TRAIN_MILITARY London 300",
            "NAVAL_MISSION 0 PATROL", "STRATEGIC_BOMB Rouen",
            "CAS_SUPPORT Dover", "BUILD_SHIP CORVETTE",
            "BUILD_SQUADRON AIR_SUPERIORITY", "PLANT_SPY Paris",
            "CODE_BREAK", "IMPOSE_SANCTIONS 0.5", "NOOP",
        ]
        n = rng.integers(1, 4)
        chosen = rng.choice(actions, size=n, replace=False)
        return "\n".join(chosen)


# ═══════════════════════════════════════════════════════════════════════════ #
# Benchmark Runner                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

def run_benchmark(args: argparse.Namespace) -> Dict:
    """Run the full LLM benchmark game."""
    print("=" * 70)
    print("  AIR STRIP ONE — LLM Strategic Benchmark")
    print(f"  Turns: {args.turns}")
    print(f"  Mode: {'dry-run (random actions)' if args.dry_run else 'LIVE API calls'}")
    print(f"  ⚙ Oceania:  {args.oceania_model}")
    print(f"  ☭ Eurasia:  {args.eurasia_model}")
    print(f"  👻 Winston:  {args.winston_model}")
    print(f"  📻 Commentary: {args.commentary_model}")
    print("=" * 70)
    print()

    rng = np.random.default_rng(args.seed)
    game = create_game(seed=args.seed, max_turns=args.turns)

    oceania_llm = LLMClient(model=args.oceania_model, dry_run=args.dry_run, max_tokens=1024, label="Oceania")
    eurasia_llm = LLMClient(model=args.eurasia_model, dry_run=args.dry_run or args.oceania_only, max_tokens=1024, label="Eurasia")
    winston_llm = LLMClient(model=args.winston_model, dry_run=args.dry_run, max_tokens=1024, label="Winston")
    commentary_llm = LLMClient(model=args.commentary_model, dry_run=args.dry_run or args.no_commentary, max_tokens=4096, label="Commentary")

    turn_log = []
    t_start = time.time()

    for turn in range(args.turns):
        if game.game_over:
            break

        print(f"\n{'─' * 60}")
        print(f"  TURN {turn + 1}/{args.turns}")
        print(f"{'─' * 60}")

        # ── Prepare summaries (instant — no API calls) ──────────────── #
        summary_0 = summarize_turn(game, faction_id=0)
        summary_1 = summarize_turn(game, faction_id=1)
        winston_active = (game.resistance and game.resistance.winston.is_alive
                          and not game.resistance.winston.is_captured)
        blf_summary = summarize_blf_turn(game) if winston_active else ""

        # ── Build turn history context (budget saver) ────────────────── #
        # Condensed summary of older turns + full detail for last 5 turns
        # Saves ~50-60% tokens vs repeating full state every call
        DETAIL_TURNS = 5
        history_lines = []
        if len(turn_log) > DETAIL_TURNS:
            # Condensed summary of old turns (1 line each)
            history_lines.append("=== PREVIOUS TURNS (condensed) ===")
            for entry in turn_log[:-DETAIL_TURNS]:
                t = entry["turn"]
                scores = entry.get("scores", {})
                o_s = int(scores.get("0", scores.get(0, 0)))
                e_s = int(scores.get("1", scores.get(1, 0)))
                # Show just the action results as a brief line
                o_acts = "; ".join(entry.get("oceania_results", [])[:2])
                e_acts = "; ".join(entry.get("eurasia_results", [])[:2])
                history_lines.append(f"T{t}: O={o_s} E={e_s} | O:[{o_acts[:80]}] E:[{e_acts[:80]}]")
            history_lines.append("")

        if len(turn_log) > 0:
            # Full detail for last 5 turns
            recent = turn_log[-DETAIL_TURNS:]
            if recent:
                history_lines.append("=== RECENT TURNS (full detail) ===")
                for entry in recent:
                    t = entry["turn"]
                    history_lines.append(f"--- Turn {t} ---")
                    history_lines.append(f"  Your actions: {'; '.join(entry.get('oceania_results', []))}")
                    history_lines.append(f"  Enemy actions: {'; '.join(entry.get('eurasia_results', []))}")
                    if entry.get("winston_results"):
                        history_lines.append(f"  BLF: {'; '.join(entry['winston_results'])}")
                history_lines.append("")

        history_block = "\n".join(history_lines) if history_lines else ""

        # Build Eurasia's history (swap perspective)
        e_history_lines = []
        if len(turn_log) > DETAIL_TURNS:
            e_history_lines.append("=== PREVIOUS TURNS (condensed) ===")
            for entry in turn_log[:-DETAIL_TURNS]:
                t = entry["turn"]
                scores = entry.get("scores", {})
                o_s = int(scores.get("0", scores.get(0, 0)))
                e_s = int(scores.get("1", scores.get(1, 0)))
                e_acts = "; ".join(entry.get("eurasia_results", [])[:2])
                o_acts = "; ".join(entry.get("oceania_results", [])[:2])
                e_history_lines.append(f"T{t}: E={e_s} O={o_s} | You:[{e_acts[:80]}] Enemy:[{o_acts[:80]}]")
            e_history_lines.append("")
        if len(turn_log) > 0:
            recent = turn_log[-DETAIL_TURNS:]
            if recent:
                e_history_lines.append("=== RECENT TURNS (full detail) ===")
                for entry in recent:
                    t = entry["turn"]
                    e_history_lines.append(f"--- Turn {t} ---")
                    e_history_lines.append(f"  Your actions: {'; '.join(entry.get('eurasia_results', []))}")
                    e_history_lines.append(f"  Enemy actions: {'; '.join(entry.get('oceania_results', []))}")
                    if entry.get("winston_results"):
                        e_history_lines.append(f"  BLF: {'; '.join(entry['winston_results'])}")
                e_history_lines.append("")
        e_history_block = "\n".join(e_history_lines) if e_history_lines else ""

        # Prepend history to summaries
        full_summary_0 = history_block + summary_0 if history_block else summary_0
        full_summary_1 = e_history_block + summary_1 if e_history_block else summary_1

        if args.verbose:
            print(f"\n[Oceania Briefing]\n{full_summary_0[:500]}...")
            print(f"\n[Eurasia Briefing]\n{full_summary_1[:500]}...")
            if winston_active:
                print(f"\n[The Ghost's Briefing]\n{blf_summary[:400]}...")

        # ── PARALLEL LLM calls (3x speedup) ─────────────────────────── #
        from concurrent.futures import ThreadPoolExecutor

        t_parallel_start = time.time()

        def _call_oceania():
            return oceania_llm.chat(OCEANIA_SYSTEM_PROMPT, full_summary_0)

        def _call_eurasia():
            return eurasia_llm.chat(EURASIA_SYSTEM_PROMPT, full_summary_1)

        def _call_winston():
            if winston_active:
                return winston_llm.chat(WINSTON_SYSTEM_PROMPT, blf_summary)
            return ""

        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_o = pool.submit(_call_oceania)
            fut_e = pool.submit(_call_eurasia)
            fut_w = pool.submit(_call_winston)
            response_0 = fut_o.result()
            response_1 = fut_e.result()
            winston_response = fut_w.result()

        t_parallel = time.time() - t_parallel_start

        # ── Apply actions sequentially (order matters for game state) ── #
        actions_0 = parse_action(response_0, game, faction_id=0)
        results_0 = apply_actions(game, faction_id=0, actions=actions_0, rng=rng)
        oceania_llm.actions_parsed += len(actions_0)
        oceania_llm.actions_noop += sum(1 for a in actions_0 if a.get('type') == 'noop')

        actions_1 = parse_action(response_1, game, faction_id=1)
        results_1 = apply_actions(game, faction_id=1, actions=actions_1, rng=rng)
        eurasia_llm.actions_parsed += len(actions_1)
        eurasia_llm.actions_noop += sum(1 for a in actions_1 if a.get('type') == 'noop')

        winston_results = []
        if winston_active and winston_response:
            blf_actions = parse_blf_action(winston_response, game)
            winston_results = apply_blf_actions(game, blf_actions, rng)
            winston_llm.actions_parsed += len(blf_actions)

        # ── Print results ─────────────────────────────────────────────── #
        print(f"  ⚙ Oceania + ☭ Eurasia + 👻 Winston (parallel {t_parallel:.1f}s):")
        for r in results_0:
            print(f"    [O] {r}")
        for r in results_1:
            print(f"    [E] {r}")
        for r in winston_results:
            print(f"    [W] {r}")

        # ── Step the game ─────────────────────────────────────────────── #
        feedback = step_game(game, rng)

        print(f"\n  Score: Oceania {game.faction_scores[0]:.0f} | Eurasia {game.faction_scores[1]:.0f}")

        # ── War Correspondent Commentary ──────────────────────────────── #
        commentary_text = ""
        visible_text = ""
        if not args.no_commentary:
            visible_text = generate_visible_events(game, feedback)
            commentary_prompt = format_commentary_prompt(visible_text)
            # Rate limit: add 1s delay if last commentary was slow (429 likely)
            if commentary_llm.latencies and commentary_llm.latencies[-1] > 12.0:
                time.sleep(2.0)
            t0 = time.time()
            dispatch = commentary_llm.chat(COMMENTARY_SYSTEM_PROMPT, commentary_prompt)
            t_comm = time.time() - t0
            # Trim to last complete sentence (don't truncate mid-word)
            raw = dispatch.strip()
            if len(raw) > 800:
                for end_char in ['. ', '." ', '.\n', '."', '.\'']:
                    idx = raw[:800].rfind(end_char)
                    if idx > 200:
                        raw = raw[:idx + 1]
                        break
                else:
                    raw = raw[:800]
            commentary_text = raw
            print(f"\n  📻 WAR DISPATCH ({t_comm:.1f}s):")
            print(f"  ┃ {commentary_text}")

        # Log
        turn_log.append({
            "turn": turn + 1,
            "oceania_actions": response_0,
            "eurasia_actions": response_1,
            "winston_actions": winston_response,
            "oceania_results": results_0,
            "eurasia_results": results_1,
            "winston_results": winston_results,
            "scores": dict(game.faction_scores),
            "naval_battles": feedback.get("naval", {}).get("total_battles", 0),
            "air_battles": feedback.get("air", {}).get("air_battles", 0),
            "blf_escalation": feedback.get("resistance", {}).get("escalation", 0),
            "blf_members": feedback.get("resistance", {}).get("total_members", 0),
            "commentary": commentary_text,
            "visible_events": visible_text,
        })

    # ── Final Results ─────────────────────────────────────────────────── #
    elapsed = time.time() - t_start
    winner = max(game.faction_scores, key=game.faction_scores.get)
    winner_name = game.faction_names[winner]

    # Collect per-faction stats
    all_clients = [
        ("Oceania", oceania_llm),
        ("Eurasia", eurasia_llm),
        ("Winston", winston_llm),
        ("Commentary", commentary_llm),
    ]
    total_tokens_in = sum(c.total_tokens_in for _, c in all_clients)
    total_tokens_out = sum(c.total_tokens_out for _, c in all_clients)
    total_calls = sum(c.total_calls for _, c in all_clients)
    total_failed = sum(c.failed_calls for _, c in all_clients)
    total_retried = sum(c.retried_calls for _, c in all_clients)

    print()
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"  Winner: {winner_name}")
    print(f"  Oceania: {game.faction_scores[0]:.0f} pts | Eurasia: {game.faction_scores[1]:.0f} pts")
    print(f"  Turns: {game.turn} | Time: {elapsed:.1f}s ({elapsed / max(game.turn, 1):.1f}s/turn)")
    print()

    # ── Per-Faction LLM Statistics ────────────────────────────────────── #
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │              LLM BENCHMARK STATISTICS                       │")
    print("  ├──────────────┬──────────┬──────────┬──────────┬────────────┤")
    print("  │ Faction      │ Model    │ Calls    │ Tokens   │ Avg Lat.   │")
    print("  ├──────────────┼──────────┼──────────┼──────────┼────────────┤")
    for name, client in all_clients:
        s = client.stats()
        model_short = s['model'][:18]
        calls_str = f"{s['successful_calls']}/{s['total_calls']}"
        tok_str = f"{s['tokens_in']+s['tokens_out']:,}"
        lat_str = f"{s['latency_avg_s']:.1f}s"
        print(f"  │ {name:<12} │ {model_short:<8} │ {calls_str:<8} │ {tok_str:<8} │ {lat_str:<10} │")
    print("  └──────────────┴──────────┴──────────┴──────────┴────────────┘")
    print()

    # ── Detailed Per-Faction Breakdown ────────────────────────────────── #
    for name, client in all_clients:
        s = client.stats()
        print(f"  {name} ({s['model']}):")
        print(f"    Calls: {s['successful_calls']} ok / {s['failed_calls']} failed / {s['retried_calls']} retried ({s['success_rate']} success)")
        print(f"    Tokens: {s['tokens_in']:,} in + {s['tokens_out']:,} out = {s['tokens_total']:,} total")
        print(f"    Per-call avg: {s['avg_tokens_per_call_in']} in / {s['avg_tokens_per_call_out']} out")
        print(f"    Latency: avg {s['latency_avg_s']}s | p50 {s['latency_p50_s']}s | p95 {s['latency_p95_s']}s | min {s['latency_min_s']}s | max {s['latency_max_s']}s")
        print()

    # ── Aggregate Totals ──────────────────────────────────────────────── #
    print(f"  TOTALS: {total_calls} calls | {total_tokens_in + total_tokens_out:,} tokens | {total_failed} failures | {total_retried} retries")
    print(f"  Cost estimate: ~${(total_tokens_in * 0.25 + total_tokens_out * 1.25) / 1_000_000:.4f} (Haiku pricing)")
    print()

    # ── Game State Summary ────────────────────────────────────────────── #
    oc_ships = len(game.naval.faction_ships(0))
    eu_ships = len(game.naval.faction_ships(1))
    oc_sqs = len(game.air.faction_squadrons(0))
    eu_sqs = len(game.air.faction_squadrons(1))
    blf_esc = game.resistance.escalation.value if game.resistance else 0
    blf_mem = game.resistance.total_members if game.resistance else 0
    winston_alive = game.resistance.winston.is_alive and not game.resistance.winston.is_captured if game.resistance else False

    oc_land = 0
    eu_land = 0
    if game.land is not None:
        for units in game.land.garrisons.values():
            for u in units:
                if u.is_alive:
                    if u.faction_id == 0:
                        oc_land += 1
                    else:
                        eu_land += 1

    print(f"  Naval: Oceania {oc_ships} ships | Eurasia {eu_ships} ships")
    print(f"  Air: Oceania {oc_sqs} squadrons | Eurasia {eu_sqs} squadrons")
    print(f"  Land: Oceania {oc_land} units | Eurasia {eu_land} units")
    print(f"  BLF: escalation {blf_esc}/4 | {blf_mem} members | Winston {'alive' if winston_alive else 'CAPTURED/DEAD'}")
    active_inv = [inv for inv in game.invasions if inv.is_active]
    if active_inv:
        print(f"  Active invasions: {len(active_inv)}")
    print("=" * 70)

    # ── Save comprehensive log ────────────────────────────────────────── #
    log_path = Path(args.log_dir) / f"benchmark_{args.seed}_{int(time.time())}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "benchmark": {
                "seed": args.seed,
                "turns_played": game.turn,
                "turns_requested": args.turns,
                "elapsed_seconds": round(elapsed, 1),
                "seconds_per_turn": round(elapsed / max(game.turn, 1), 1),
                "winner": winner_name,
                "scores": dict(game.faction_scores),
            },
            "models": {
                "oceania": args.oceania_model,
                "eurasia": args.eurasia_model,
                "winston": args.winston_model,
                "commentary": args.commentary_model,
            },
            "llm_statistics": {
                name.lower(): client.stats() for name, client in all_clients
            },
            "aggregates": {
                "total_calls": total_calls,
                "total_tokens_in": total_tokens_in,
                "total_tokens_out": total_tokens_out,
                "total_tokens": total_tokens_in + total_tokens_out,
                "total_failures": total_failed,
                "total_retries": total_retried,
                "cost_estimate_usd": round((total_tokens_in * 0.25 + total_tokens_out * 1.25) / 1_000_000, 6),
            },
            "game_state": {
                "oceania_ships": oc_ships,
                "eurasia_ships": eu_ships,
                "oceania_squadrons": oc_sqs,
                "eurasia_squadrons": eu_sqs,
                "blf_escalation": blf_esc,
                "blf_members": blf_mem,
                "winston_alive": winston_alive,
                "active_invasions": len(active_inv),
            },
            "turn_log": turn_log,
        }, f, indent=2, default=str)
    print(f"  Log saved: {log_path}")

    return {
        "winner": winner_name,
        "scores": dict(game.faction_scores),
        "turns": game.turn,
        "stats": {name.lower(): client.stats() for name, client in all_clients},
    }


# ═══════════════════════════════════════════════════════════════════════════ #
# CLI                                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def main():
    p = argparse.ArgumentParser(description="LLM Benchmark for Air Strip One")
    p.add_argument("--model", default="mistral-small-latest",
                   help="Default model (used if faction-specific not set)")
    p.add_argument("--oceania-model", default=None,
                   help="Model for Oceania (default: claude-haiku-20241022)")
    p.add_argument("--eurasia-model", default=None,
                   help="Model for Eurasia (default: open-mixtral-8x22b)")
    p.add_argument("--winston-model", default=None,
                   help="Model for Winston/BLF (default: same as oceania)")
    p.add_argument("--commentary-model", default="mistral-medium-latest",
                   help="Model for war correspondent commentary")
    p.add_argument("--turns", type=int, default=50,
                   help="Number of game turns (default: 50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="No API calls — random actions for testing")
    p.add_argument("--oceania-only", action="store_true",
                   help="Only Oceania uses LLM, Eurasia plays random")
    p.add_argument("--verbose", action="store_true",
                   help="Print full briefings")
    p.add_argument("--no-commentary", action="store_true",
                   help="Disable war correspondent commentary")
    p.add_argument("--log-dir", default="logs/llm_benchmark",
                   help="Directory for benchmark logs")
    p.add_argument("--env-file", default=None,
                   help="Path to .env file (default: project root .env)")
    args = p.parse_args()
    if args.env_file:
        _load_env(args.env_file)
    # Apply faction model defaults
    if args.oceania_model is None:
        args.oceania_model = "claude-haiku-4-5-20251001" if os.environ.get("ANTHROPIC_API_KEY") else args.model #I MADE THIS, don't change
    if args.eurasia_model is None:
        args.eurasia_model = "mistral-medium-latest" if os.environ.get("MISTRAL_API_KEY") else args.model #I MADE THIS, don't change
    if args.winston_model is None:
        args.winston_model = args.oceania_model
    run_benchmark(args)


if __name__ == "__main__":
    from typing import Dict
    main()
