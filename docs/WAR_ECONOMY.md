# War Economy & Diplomacy System

> **NOTE**: This document describes the legacy Leontief economy model (`extensions/war_economy/`).
> The current system uses **Economy V2** (`extensions/economy_v2/economy_core.py`) with 10 factory types,
> GDP calculation, and power dependency. See [AIRSTRIP_ONE_SYSTEMS.md](AIRSTRIP_ONE_SYSTEMS.md) for full documentation.
>
> Additionally, the **Governance System** (`extensions/governance/budget_system.py`) handles budget distribution
> (7 categories), corruption (asymptotic floor), and bureaucracy delays.
> The **Population System** (`extensions/pop/pop_v2.py`) uses real 1958 census numbers with 1984 social classes.
> The **Research System** (`extensions/research/research_system.py`) provides a 10×5 HOI4-style tech tree with 50 technologies.

## Architecture (Legacy — runs alongside new systems)

```
extensions/war_economy/
  __init__.py                # Public API
  war_economy_state.py       # EconSector, Resource, ClusterEconomy, I/O matrices
  war_economy_dynamics.py    # Leontief production engine, trade, lend-lease
  war_economy_actions.py     # Agent actions, observation builder
```

## Economic Sectors (7)

| Sector | Produces | Consumes | Role |
|--------|----------|----------|------|
| **Agriculture** | Raw Food, Processed Food | Fuel, Chemicals | Feed population |
| **Mining** | Iron Ore, Coal, Raw Materials, Crude Oil | Fuel, Steel | Extract raw materials |
| **Energy** | Fuel | Crude Oil, Coal, Steel | Power everything |
| **Heavy Industry** | Steel, Chemicals | Iron Ore, Coal, Fuel | Process intermediates |
| **Manufacturing** | Consumer Goods, Military Equipment, Ammunition | Steel, Chemicals, Fuel | Produce finished goods |
| **Construction** | Infrastructure repair + capacity maintenance | Steel, Fuel, Raw Materials | Prevent decay |
| **Services** | Productivity boost | Processed Food, Consumer Goods | Boost all sectors |

Sectors are interconnected via the **Leontief input-output matrix**. Each sector requires specific inputs to produce output. A shortage of ANY input bottlenecks the entire sector.

## Resources — 3-Tier Supply Chain (12 types)

### Tier 1 — Raw (extracted from terrain)
| Resource | Source Terrain | Primary Sector |
|----------|---------------|----------------|
| Crude Oil | Desert, Marsh | Mining |
| Iron Ore | Mountains | Mining |
| Coal | Mountains, Urban | Mining |
| Raw Food | Plains | Agriculture |
| Raw Materials | Forest (timber, rubber) | Mining |

### Tier 2 — Processed (from Tier 1)
| Resource | Made From | Sector |
|----------|-----------|--------|
| Fuel | Crude Oil + Coal | Energy |
| Steel | Iron Ore + Coal | Heavy Industry |
| Chemicals | Coal + Raw Materials | Heavy Industry |
| Processed Food | Raw Food | Agriculture |

### Tier 3 — Finished (from Tier 2)
| Resource | Made From | Sector |
|----------|-----------|--------|
| Consumer Goods | Steel + Chemicals + Processed Food | Manufacturing |
| Military Equipment | Steel + Chemicals + Fuel | Manufacturing |
| Ammunition | Chemicals + Steel | Manufacturing |

**Key constraint**: You cannot produce Tier 3 without Tier 2, which requires Tier 1. Bottleneck at any tier cascades upward.

## Leontief Production Model

```
For each sector s in a cluster:
  input_ratio[r] = stockpile[r] / SECTOR_INPUTS[s, r]
  bottleneck = min(input_ratio) over all required inputs
  labor_eff = 1 - exp(-K × labor[s] / capacity[s])     # diminishing returns
  output = effective_capacity[s] × labor_eff × min(bottleneck, capacity)
         × (1 - sanctions) × (1 - war_fatigue) × war_bond_mult
```

Processing follows tier order: Mining/Agriculture first, then Energy/Heavy Industry, then Manufacturing. This ensures upstream resources are available before downstream sectors try to use them.

## Anti-Exploitation Mechanisms

| Mechanism | Effect | Why It Matters |
|-----------|--------|----------------|
| **Leontief bottleneck** | Output = min(available/required) for ALL inputs | Can't cheese by stockpiling one resource |
| **Diminishing returns** | `1 - exp(-K × labor/cap)` | More workers past capacity gives minimal benefit |
| **Capital depreciation** | 1.5%/step without Construction | Must invest in maintenance or lose capacity |
| **Infrastructure decay** | 0.5%/step without Construction | ALL sectors degrade if Construction neglected |
| **Labor rigidity** | Max 10% workforce shift per step | Can't instantly pivot economy |
| **Spoilage** | Food decays 0.8%/step, chemicals 0.4%/step | Can't hoard perishables indefinitely |
| **War fatigue** | 0.1%/step productivity loss during conflict | Prolonged war grinds down economy |
| **Transition friction** | 20% waste when reallocating capital | Restructuring has real costs |
| **War damage** | Hazard > 0.3 damages capacity + infrastructure | Combat zones have economic costs |

## Agent Actions (10)

| Action | Parameters | Effect |
|--------|------------|--------|
| `REALLOCATE_LABOR` | cluster, sector_from, sector_to, intensity | Shift workforce (max 10%/step) |
| `TRADE_PROPOSE` | target_faction, resource, intensity | Bilateral trade deal |
| `LEND_LEASE` | target_faction, resource, intensity | Asymmetric military aid |
| `IMPOSE_SANCTIONS` | target_faction, intensity | Block enemy trade (up to 60% production loss) |
| `LIFT_SANCTIONS` | target_faction | Diplomatic gesture |
| `ISSUE_WAR_BONDS` | cluster | +50% production for 30 steps, accumulates debt |
| `BLOCKADE` | cluster | Block trade routes through enemy cluster |
| `SET_MFG_PRIORITY` | intensity | 0=consumer goods, 1=military equipment |
| `MOBILIZE_ECONOMY` | intensity | Faction-wide war mobilization level |

## Cross-System Feedback

### War Economy → Military
- **Supply refill** × min(fuel, food, ammo) stockpile ratios
- **Combat effectiveness** × (0.5 + 0.3×ammo + 0.2×fuel)
- **Military production** from military_equipment + ammunition stockpiles

### War Economy → Core Economy (GDP/Unemployment/Debt/Industry)
- **GDP modifier** from food + steel stockpile levels
- **Infrastructure** level affects existing Industry variable

### War Economy → Population
- **Morale** from food + consumer goods availability
- **Attrition** when food stockpile < 30%

## Observation Space

Per cluster (30 floats × max_clusters):
- 12 resource stockpile ratios
- 7 sector output levels (normalized)
- 7 sector labor fractions
- Infrastructure, war damage, sanctions, blockade

Faction-level (18 floats):
- war_mobilization, fiscal_debt, inflation, trade_balance, war_fatigue, mfg_priority
- 12 market prices

**Total**: `30 × max_clusters + 18` (default: 378 for 12 clusters)
