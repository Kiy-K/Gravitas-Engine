# Ministries System

Autonomous government ministries for the Air Strip One scenario. Each faction runs 7 ministries that operate semi-independently within their budget allocation — handling routine tasks, reporting problems to the LLM player, and requesting more funding when needed.

## Overview

The ministries layer sits between the player's high-level budget decisions and the underlying simulation systems. It:

1. Receives budget from the player's `SET_BUDGET` action each turn
2. Autonomously processes routine tasks (repairing ships, distributing food, patrolling for BLF cells, etc.)
3. Reports status back to the LLM turn summary
4. Flags urgent problems and budget requests so the player can override or redirect

This means the player doesn't have to micromanage every repair or food redistribution — the ministries handle it, degrading gracefully when underfunded.

## Ministry Roster

### Oceania (Ingsoc names)

| Ministry | Budget Category | Routine Tasks |
|----------|----------------|---------------|
| **Ministry of Peace** (Minipax) | MILITARY | Auto-repair ships below 60% HP; refit weak air squadrons |
| **Ministry of Plenty** (Miniplenty) | PRODUCTION | Redistribute food from surplus to starving sectors |
| **Ministry of Truth** (Minitrue) | WELFARE | Telescreen propaganda — small trust boost; flag low-trust sectors |
| **Ministry of Love** (Miniluv) | POLICE | Thought Police patrols; raise BLF cell detection risk |
| **Ministry of Science** | RESEARCH | Report active projects, flag idle research slots |
| **Ministry of Construction** | INFRASTRUCTURE | Auto-repair damaged factories; report power shortages |
| **Ministry of Labour** | DEBT_SERVICE | Manage training pipeline; report unemployment |
| **Anti-Corruption Agency** | POLICE (shared) | Hidden investigation — reduces corruption when complete |

### Eurasia (Soviet-style names)

| Ministry | Eurasia Name |
|----------|-------------|
| Ministry of Peace | People's Commissariat of Defense |
| Ministry of Plenty | State Planning Committee (Gosplan) |
| Ministry of Truth | Propaganda Directorate |
| Ministry of Love | Committee for State Security |
| Ministry of Science | Academy of Sciences |
| Ministry of Construction | Construction Committee |
| Ministry of Labour | Commissariat of Labour |
| Anti-Corruption Agency | Anti-Corruption Inspectorate |

## Base Efficiency by Faction

| Ministry | Oceania | Eurasia | Notes |
|----------|---------|---------|-------|
| Peace | 0.85 | 0.80 | Eurasia: committee approvals slow military orders |
| Plenty | 0.65 | 0.60 | Both: endemic food distribution failures |
| Truth | 0.90 | 0.75 | Oceania telescreen network is mature |
| Love | 0.95 | 0.85 | Ingsoc Thought Police is near-perfect |
| Science | 0.60 | 0.65 | Oceania suppresses innovation; Eurasia prizes it |
| Construction | 0.70 | 0.65 | Both struggle with supply chains |
| Labour | 0.80 | 0.85 | Eurasia's collectivist system manages labor well |
| Anti-Corruption | 0.50 | 0.55 | Oceania: corruption IS the system |

## Budget Flow

```
Player: SET_BUDGET Military:35% Production:20% Research:10% ...
          │
          ▼
Governance system converts percentages → GDP amounts
          │
          ▼
Each ministry receives its GDP share
  raw_budget = allocation[category] × gdp_revenue
  effective_budget = raw_budget × (1 − corruption_rate)
          │
          ▼
Ministry performs tasks up to budget limit
  efficiency decays if underfunded (−0.02/turn below threshold)
  efficiency recovers if adequately funded (+0.01/turn)
```

## Anti-Corruption Agency — Hidden Investigation

The Anti-Corruption Agency is unique: its investigation progress bar is **invisible to the LLM player**. The agency reports only vague status ("Investigation ongoing. Following leads.").

Progress depends on:
- Funding (more POLICE budget = faster)
- Efficiency (decays if underfunded)
- Current corruption level (higher corruption = harder to investigate)
- Random luck: `rng.uniform(0.3, 2.0)` multiplier each turn
- Dead ends: 5% chance of −10% progress setback per turn

When the bar reaches 1.0, corruption is reduced by `0.02–0.08%` (scaled by funding and efficiency), then the bar resets. The player cannot game the timing because the bar is hidden and the random factor is large.

## Ministry Reports in Turn Summary

Each turn the LLM sees condensed ministry status:

```
MINISTRY STATUS:
  Ministry of Peace: 2 tasks done
  Ministry of Plenty: UNDERFUNDED, 1 issues
  Ministry of Truth: operating normally
  Ministry of Love: REQUESTING MORE FUNDS
  Ministry of Science: 1 issues
  Ministry of Construction: 2 tasks done
  Ministry of Labour: operating normally
  URGENT ISSUES:
    ⚠ 4 sectors below 15% food — increase WELFARE spending!
    ⚠ BLF at level 3 — increase POLICE budget!
  BUDGET REQUESTS:
    Ministry of Plenty: FOOD CRISIS: 4 sectors starving!
```

## Task Capacity

Each ministry can hold at most **3 active tasks** at once. When at capacity, new auto-tasks are skipped until existing tasks complete. Each task has a `turns_remaining` countdown; `budget_spent` is distributed across turns.

## Source

`extensions/ministries/ministries.py` — `MinistryType`, `Ministry`, `FactionMinistries`, `MinistryWorld`, `step_ministries()`, `ministry_reports()`
