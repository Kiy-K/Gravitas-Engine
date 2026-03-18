# Air Strip One — Complete Systems Documentation

## Overview

**GRAVITAS Engine: Air Strip One** is a strategic war simulation set in George Orwell's 1984 universe. Three AI players (Oceania, Eurasia, Winston Smith / BLF resistance) compete in a turn-based game where each turn represents **1 week** of in-game time. A fourth LLM provides war correspondent narrative commentary.

The simulation integrates 13 interconnected systems across a 35-sector map covering all of the British Isles, France, Benelux, and Netherlands.

---

## 1. Map — 35 Sectors, 6 Sea Zones

### Oceania (18 sectors — All British Isles)

| ID | City | Pop (1958) | Role | Key Assets |
|----|------|-----------|------|------------|
| 0 | London | 7,500K | Capital | Ingsoc HQ, Ministries, 85% proles |
| 1 | Dover | 25K | Garrison | Channel fortress, radar, AA |
| 2 | Portsmouth | 247K | Naval Base | Home Fleet HQ, shipyards |
| 3 | Southampton | 180K | Port City | Factory district, supply port |
| 4 | Canterbury | 30K | Airbase | RAF 11 Group, training camps |
| 5 | Brighton | 150K | Garrison | Secondary coastal defense |
| 6 | Bristol | 420K | Industrial | Aircraft factories, western hub |
| 7 | Plymouth | 113K | Naval Base | Western Approaches HQ |
| 8 | Cardiff | 250K | Industrial | Welsh coal + steel |
| 9 | Birmingham | 970K | Industrial | Tank production, Midlands heavy industry |
| 10 | Manchester | 640K | Industrial | Munitions, textiles → uniforms |
| 11 | Liverpool | 700K | Port City | Atlantic convoy port |
| 12 | Leeds | 480K | Logistics Hub | Northern industry + rail junction |
| 13 | Norwich | 130K | Airbase | RAF Bomber Command, agricultural |
| 14 | Edinburgh | 470K | Naval Base | Forth anchorage, Scottish Command |
| 15 | Glasgow | 900K | Industrial | Clyde shipyards, heavy industry |
| 16 | Dublin | 650K | Port City | Airstrip Two capital, Atlantic Fleet |
| 17 | Belfast | 440K | Naval Base | Harland & Wolff shipyards |

### Eurasia (17 sectors — France + Benelux + Netherlands)

| ID | City | Pop (1958) | Role | Key Assets |
|----|------|-----------|------|------------|
| 18 | Calais | 75K | Garrison | Channel invasion staging, artillery |
| 19 | Dunkirk | 25K | Naval Base | Channel Fleet base |
| 20 | Le Havre | 140K | Port City | Normandy port (rebuilt post-WWII) |
| 21 | Cherbourg | 35K | Sub Base | Submarine pens, Atlantic approach |
| 22 | Amiens | 110K | Logistics Hub | Picardy rail junction |
| 23 | Rouen | 115K | Industrial | Seine steel mills |
| 24 | Lille | 195K | Industrial | Nord coal + reserves |
| 25 | Brussels | 1,100K | Capital | Benelux Command HQ |
| 26 | Antwerp | 260K | Naval Base | Major port, North Sea fleet |
| 27 | Rotterdam | 750K | Naval Base | Europoort, North Sea Fleet Pride |
| 28 | Amsterdam | 870K | Trade Hub | Trade, finance, industry |
| 29 | Luxembourg | 200K | Industrial | Steel industry, quiet rear area |
| 30 | Paris | 5,000K | Capital | Western Front Command HQ |
| 31 | Orleans | 85K | Logistics Hub | Loire crossing, central logistics |
| 32 | Lyon | 530K | Industrial | Rhône-Alpes industry |
| 33 | Brest | 120K | Sub Base | Finistère submarine pens |
| 34 | Bordeaux | 250K | Default | Southern reserves |

### Sea Zones (6)

| Zone | Name | Width | Connects | Key Threat |
|------|------|-------|----------|------------|
| 0 | Dover Strait | 33km | Dover ↔ Calais | PRIMARY — shortest crossing |
| 1 | Western Channel | 120km | Portsmouth/Brighton ↔ Dunkirk/Le Havre | Thames Estuary + Normandy route |
| 2 | North Sea | 500km | Liverpool/Leeds/Edinburgh ↔ Antwerp/Rotterdam/Amsterdam | Flanking via expanded Benelux fleet |
| 3 | Irish Sea | 300km | Dublin/Belfast ↔ Liverpool/Glasgow | Wolf packs starve Ireland |
| 4 | Bay of Biscay | 600km | Plymouth ↔ Brest/Bordeaux/Cherbourg | Brest submarine pens |
| 5 | North Atlantic | Open ocean | Dublin/Glasgow ↔ convoy routes | Convoy lifeline, Brest subs |

Population data sourced from **1958 census** interpolation (1951/1961 UK Census, 1954 French Census).

---

## 2. Economy System (`extensions/economy_v2/economy_core.py`)

### 10 Factory Types per Cluster

| Type | Purpose | Power Required | GDP Value |
|------|---------|---------------|-----------|
| POWER_PLANT | Electricity — enables ALL factories | No | 0.8 |
| CIVIL_FACTORY | Consumer goods → GDP + morale | Yes | 1.5 |
| MIL_FACTORY | Arms, ammo, vehicles, equipment | Yes | 2.0 |
| DOCKYARD | Ship production (slow) | Yes | 2.5 |
| AIRFIELD | Aircraft assembly + maintenance | Yes | 1.8 |
| STEEL_MILL | Raw ore → steel (feeds MIL + DOCKYARD) | Yes | 1.2 |
| REFINERY | Crude → fuel (lifeblood of mech war) | Yes | 1.6 |
| FARM | Food production | **No** (only exception) | 0.6 |
| HOSPITAL | Medical care, heals wounded | Yes | 0.5 |
| INFRASTRUCTURE | Roads, rail, ports → logistics | Yes | 0.4 |

**Factory Output** = `level × (1 - damage) × staffing_ratio × efficiency × power_ratio`

**Power Dependency**: All factories except FARM need power. If power_ratio < 1.0, all output is reduced proportionally. This prevents exploit of just building military factories — you NEED power plants.

### GDP Model
- **GDP** = Σ(factory_output × GDP_value) across all clusters
- **GDP per capita** = GDP / (population / 10000)
- **War economy ratio**, inflation, debt, war weariness tracked per faction

### Cluster Roles
Each cluster has a factory profile matching its historical role:
- **capital**: London, Paris, Brussels — high civil + infrastructure
- **industrial**: Birmingham, Cardiff, Glasgow, Rouen, Lyon, Lille — heavy MIL + STEEL
- **naval_base**: Portsmouth, Plymouth, Edinburgh, Belfast, Dunkirk, Antwerp — DOCKYARD focus
- **airbase**: Norwich, Canterbury — AIRFIELD focus
- **port_city**: Liverpool, Southampton, Le Havre — DOCKYARD + REFINERY
- **sub_base**: Brest, Cherbourg — DOCKYARD + REFINERY
- **garrison**: Dover, Brighton, Calais — MIL_FACTORY + HOSPITAL
- **logistics_hub**: Leeds, Amiens, Orleans — INFRASTRUCTURE heavy
- **agricultural**: (default for rural areas) — FARM focus

---

## 3. Population System (`extensions/pop/pop_v2.py`)

### Real Numbers, Not [0,1]

London has **7,500,000 people**. Dover has **25,000**. Conscripting 10,000 from London is 0.1% — from Dover it's 40%. The math matters.

### 1984 Social Classes

| Class | % of Pop | Political Power / Capita | Income | Role |
|-------|----------|------------------------|--------|------|
| **Inner Party** | 2% | 50.0 | 5.0× | O'Brien's class. Run the Ministries. |
| **Outer Party** | 13% | 3.0 | 1.2× | Winston's former class. Bureaucrats. Surveilled. |
| **Proles** | 85% | 0.2 | 0.4× | The masses. "Proles and animals are free." |

### 8 Job Categories

FARMER, FACTORY_WORKER, MINER, SOLDIER, BUREAUCRAT, TECHNICIAN, POLICE, UNEMPLOYED

Job distribution varies by cluster role (industrial cities have more factory workers, garrison cities have more soldiers, etc.)

### Conscription Scaling

| Law | % of Working Age Drafted |
|-----|-------------------------|
| VOLUNTEER_ONLY | 2% |
| LIMITED_DRAFT | 5% |
| GENERAL_MOBILISATION | 12% |
| TOTAL_MOBILISATION | 25% |
| SCRAPING_THE_BARREL | 40% |

Conscription draws from: **unemployed proles first** → **unemployed outer party** → **employed prole factory workers** (with GDP penalty). **Inner Party is EXEMPT.**

---

## 4. Research System (`extensions/research/research_system.py`)

### 10 Branches × 5 Tiers = 50 Technologies

| Branch | Focus | T5 Capstone |
|--------|-------|-------------|
| INDUSTRY | Factory output, construction, resources | Nuclear Power (needs Electronics T3) |
| ELECTRONICS | Radar, code-breaking, computing | Computing Machines |
| NAVAL | Ship combat, submarines, torpedoes | Nuclear Propulsion (needs Industry T4) |
| AIR | Fighters, bombing, speed | Jet Propulsion (needs Industry T3) |
| LAND | Infantry, artillery, armor, forts | Guided Munitions (needs Electronics T3) |
| DOCTRINE | Org recovery, supply, planning | Total War Doctrine |
| NUCLEAR | Atomic weapons, power, radiation | Thermonuclear Weapons (needs Industry T5) |
| ROCKETRY | Ballistic missiles, space, delivery | ICBM (needs Electronics T4) |
| CRYPTOGRAPHY | Encryption, signals intelligence | Quantum Cryptography (needs Electronics T5) |
| INFRASTRUCTURE | Roads, rail, ports, logistics | Maglev Networks (needs Industry T4) |

**Research Time**: T1=8 weeks, T2=12, T3=18, T4=25, T5=35 weeks

**Cross-Branch Prerequisites** (prevents rushing top-tier):
- Nuclear Power → needs Electronics T3
- Nuclear Submarine → needs Industry T4
- Jet Propulsion → needs Industry T3
- Guided Munitions → needs Electronics T3
- Advanced Sonar → needs Electronics T2
- Thermonuclear Weapons → needs Industry T5 + Nuclear T4
- ICBM → needs Rocketry T4 + Electronics T4
- Quantum Cryptography → needs Electronics T5 + Cryptography T4
- Maglev Networks → needs Infrastructure T4 + Industry T4

**Anti-Exploitation**: Max 2 simultaneous projects. Bonuses are multiplicative and incremental (+8-12% per tier, not +20%). Can't skip tiers.

---

## 5. Governance System (`extensions/governance/budget_system.py`)

### Budget Distribution (7 Categories)

| Category | Min | Purpose | Anti-Exploit |
|----------|-----|---------|-------------|
| MILITARY | 5% (max 60%) | Combat, equipment, maintenance | Cap at 60% — above that, economy collapses |
| PRODUCTION | 5% | Factory output boost | Diminishing returns above 30% |
| RESEARCH | 3% | Tech speed multiplier | Below 5% = NO research progress |
| WELFARE | 5% | Prole satisfaction, food | Below 10% = starvation + unrest |
| POLICE | 3% | Thought Police, resistance suppression | Below 5% = BLF grows unchecked |
| INFRASTRUCTURE | 3% | Logistics, bureaucracy speed | Affects ALL other systems |
| DEBT_SERVICE | 1% | Pay down war debt | Reduces inflation |

**Budget constraints**: Must sum to 100%. Max ±15% change per turn. Diminishing returns above 30% (sqrt curve).

### Corruption System

**Asymptotic Design**: Corruption can NEVER reach 0%.

| Faction | Base Rate | Floor (minimum) |
|---------|-----------|-----------------|
| Oceania | 8% | 3.2% |
| Eurasia | 12% | 4.8% |

**Corruption grows from**: High military spending (procurement fraud), high police (extortion), low welfare (desperate officials), war duration (+0.8%/week)

**Corruption shrinks from**: ANTI_CORRUPTION action (-3% but diminishing returns), high welfare, Electronics research

**Effects**: Effective budget = allocated × (1 - corruption_rate). Slows GDP, reduces equipment quality, erodes trust.

### Bureaucracy Delays

Every non-instant action goes through a paperwork pipeline:

| Speed | Delay | Examples |
|-------|-------|---------|
| Instant (0 weeks) | Field commands | NAVAL_MISSION, CAS_SUPPORT, LAY_MINES |
| Fast (1 week) | Standing orders | STRATEGIC_BOMB, PLANT_SPY, CODE_BREAK |
| Medium (2 weeks) | Procurement | BUILD_SHIP, BUILD_SQUADRON, CONSCRIPT |
| Slow (3 weeks) | Policy changes | BUILD_FACTORY, SET_BUDGET, ANTI_CORRUPTION |
| Very slow (4 weeks) | Committee | RESEARCH |

**Delay modifiers**: Corruption adds random +1-2 weeks (lost paperwork). Infrastructure budget reduces delays. Regime speed: Oceania 1.15× (totalitarian), Eurasia 0.85× (committees). Frequent budget changes add +2 week penalty.

---

## 6. Naval System (`extensions/naval/`)

### 14 Ship Classes

BATTLESHIP, BATTLECRUISER, HEAVY_CRUISER, LIGHT_CRUISER, DESTROYER, DESTROYER_ESCORT, CORVETTE, FLEET_SUBMARINE, COASTAL_SUBMARINE, TRANSPORT, SUPPLY_SHIP, MINELAYER, FLYING_BOAT

### Fleet Compositions

**Oceania (59 ships across 6 zones)**:
- Home Fleet (Zone 1): Battleship + battlecruiser + heavy cruiser + escorts
- Dover Squadron (Zone 0): Heavy cruiser + destroyers + corvettes + minelayer
- North Sea Patrol (Zone 2): Light cruiser + destroyers + corvettes
- Atlantic Fleet (Zone 3): Battlecruiser + light cruiser + destroyers
- Plymouth Force (Zone 4): Light cruiser + destroyers + corvettes
- Atlantic Convoy Escort (Zone 5): Light cruiser + escort destroyers + corvettes

**Eurasia (45 ships across 6 zones)**:
- Calais Flotilla (Zone 0): Fleet subs + coastal subs + destroyer + minelayer
- Dunkirk Fleet (Zone 1): Heavy cruiser + destroyers + subs + 3 TRANSPORTS
- Baltic Fleet Det. (Zone 2): Heavy cruiser + destroyers + fleet subs + minelayer
- Brest Wolf Packs (Zone 4): Heavy cruiser + 4 fleet subs + 2 coastal subs
- Atlantic Raiders (Zone 5): Fleet subs
- Irish Sea (Zone 3): Fleet sub + coastal sub

### 3 Invasion Types

| Type | Planning | Casualties | Best For |
|------|----------|-----------|---------|
| PREPARED | 10 weeks | Lowest (best intel) | Main invasion with naval superiority |
| RECKLESS | 3 weeks | +50% penalty | Desperate gamble, race against time |
| AIRBORNE | 5 weeks | Very high (20-70%) | Behind enemy lines, no ships needed |

---

## 7. Air Force System (`extensions/air_force/`)

### 10 Aircraft Types
INTERCEPTOR, AIR_SUPERIORITY, HEAVY_FIGHTER, TACTICAL_BOMBER, STRATEGIC_BOMBER, DIVE_BOMBER, GROUND_ATTACK, RECON_AIRCRAFT, TRANSPORT_AIRCRAFT, FLYING_BOAT

**Oceania**: ~301 squadrons across 7 bases (London, Portsmouth, Canterbury, Birmingham, Norwich, Edinburgh, Dublin)
**Eurasia**: ~280 squadrons across 7 bases (Paris, Rouen, Calais, Brussels, Antwerp, Brest, Lyon)

---

## 8. BLF Resistance System (`extensions/resistance/resistance.py`)

### 7 Escalation Levels

| Level | Name | Trigger | Effect |
|-------|------|---------|--------|
| 0 | DORMANT | Start | No effect |
| 1 | WHISPERS | Auto (BLF exists) | Graffiti, +2% polarization |
| 2 | SABOTAGE | 5+ turns at L1, 50+ members | -5% factory output, +3% hazard |
| 3 | ORGANIZED_RESISTANCE | 10+ turns at L2, 80+ members, arms | -10% industry, -10% military diversion |
| 4 | OPEN_REVOLT | 10+ turns at L3, 150+ members, food crisis/beachhead | -25% industry, -30% military diversion |
| 5 | FULL_REVOLUTION | 8+ turns at L4, 200+ members, 5+ arms, Winston alive | -50% industry, -50% military, London seized |
| 6 | BETRAYED_REVOLUTION | Eurasia declares war on BLF | -35% industry, BLF fights both sides |

### Winston Smith — The Ghost of London

Winston leads the BLF from the sewers. He can RECRUIT, STEAL_ARMS, SABOTAGE, BROADCAST (pirate telescreen), COORDINATE cells, MOVE between cities, and CONTACT_EURASIA.

**Exploit Mechanic**: Winston reads Oceania's CONDITIONS each turn:
- **Hungry/starving cities** → easier recruitment
- **Bombed cities** → easier arms raids (guards distracted)
- **"Party crumbling" cities** → easier sabotage
- **Low Oceania welfare budget** → recruitment easier everywhere

### Winston Capture Event
When caught by Thought Police: **-40% morale** to all cells, propaganda collapses (-30%), arms caches raided (-2). Oceania sees "🚨 VICTORY: WINSTON SMITH CAPTURED!" with tactical advice. Eurasia sees "BLF LEADER CAPTURED — invasion window closing."

### Eurasia Decision System
When BLF becomes dominant at FULL_REVOLUTION:
- **5-week decision window** opens for Eurasia
- **SUPPORT_BLF** — keep arming rebels (BLF strengthens)
- **DECLARE_WAR_BLF** — betray them (triggers BETRAYED_REVOLUTION)
- **Silence = auto-betrayal** when timer expires

---

## 9. Intelligence System (`extensions/intelligence/intel_system.py`)

Fog of war + espionage. Factions can only see their own sectors clearly. Enemy sectors show as [NO INTELLIGENCE], [UNCONFIRMED], or full visibility depending on:

- **Spy rings** planted via PLANT_SPY
- **Code-breaking** progress via CODE_BREAK
- **Radar** coverage (Chain Home network)
- **Adjacency** — border sectors have partial visibility

Actions: PLANT_SPY city, CODE_BREAK, COUNTER_INTEL, DECEPTION target, CHANGE_CODES

---

## 10. Weather System (`gravitas/weather_bridge.py`)

Unified land/sea/air weather per region using physics.py primitives. English Channel maritime climate with seasonal model.

**Effects**: Sea state (affects naval operations + invasions), cloud cover (affects air operations), ground conditions (mud/snow affects land combat), production (extreme weather reduces factory output).

---

## 11. War Correspondent (`COMMENTARY_SYSTEM_PROMPT`)

A fourth LLM provides narrative dispatches each turn in the style of **Edward R. Murrow** or **Ernie Pyle**. Uses `mistral-medium-latest` for richer prose.

**Fog of War**: The correspondent can only see public events — bombings, naval battles from shore, troop movements, refugee columns, propaganda, graffiti. Cannot see submarine positions, classified plans, or intelligence reports.

**Style Guide**: Start with a vivid image, end with a gut punch. Name specific 1984 locations (Victory Mansions, Chestnut Tree Café, Southwark docks). Include sensory details (cordite, oil, telescreen static). Show PEOPLE (prole woman, child collecting shrapnel, soldier writing a letter).

---

## 12. Manpower System (`extensions/war_economy/manpower.py`)

### Conscription Laws (15 levels)

| Level | Name | Draft Rate | GDP Penalty |
|-------|------|-----------|-------------|
| 0 | VOLUNTEER_ONLY | 2% working age | None |
| 1 | LIMITED_DRAFT | 5% | −2% GDP |
| 2 | GENERAL_MOBILISATION | 12% | −5% GDP |
| 3 | TOTAL_MOBILISATION | 25% | −10% GDP |
| 4 | SCRAPING_THE_BARREL | 40% | −20% GDP |

Draft pulls from **unemployed proles first** → **unemployed outer party** → **employed factory workers** (with GDP hit). Inner Party is exempt.

### Training Pipeline

Conscripts do not become soldiers instantly. They enter a multi-week training pipeline:

| Training Type | Duration | Output |
|---------------|----------|--------|
| Basic | 3 weeks | Infantry |
| Specialist | 6 weeks | Engineers, Snipers, Medics |
| Armor crew | 8 weeks | Tank crews |
| Officer | 10 weeks | Leaders (morale bonus) |

Training capacity is limited by INFRASTRUCTURE budget and available trainers. Underfunded labour ministry (via Ministries system) slows throughput.

### Reserves

Discharged veterans enter a **reserve pool**. They can be recalled to active duty faster than raw conscripts (1 week vs. training pipeline). Reserve size depends on war duration and previous mobilisation levels.

---

## 13. Land Combat System (`extensions/military/land_bridge.py`)

### CoW Unit Integration
Reuses the existing Call of War combat system from `extensions/military/cow_combat.py` with 30+ unit types:

- **Infantry**: Infantry, Militia, Guards, Ski Troops, Shock Troops, Engineers, Snipers
- **Armor**: Light Tank, Medium Tank, Heavy Tank, Tank Destroyer, Flame Tank
- **Support**: Artillery, Mortar, Rocket Artillery, Anti-Air, Supply Truck
- **Recon**: Armored Car, Recon Infantry

### Per-Sector Garrisons
Each sector can host land units from multiple factions. Combat resolves automatically each turn in contested sectors:

- **Starting forces**: 69 Oceania units + 55 Eurasia units across 35 sectors
- **Historical placement**: Guards in London, Tanks in Calais, etc.
- **Beachhead spawning**: Successful invasions spawn 2 infantry + 1 militia in target sector

### Combat Resolution
- **Lanchester laws** with terrain bonuses
- **Urban**: +35% defense for Guards
- **Plains/Open**: Standard combat
- **Contested sectors**: Show as yellow on map, resolve each turn

---

## 14. LLM Benchmark (`tests/benchmark_llm.py`)

### Multi-Provider Support
- **Anthropic** (Claude): Auto-detected from model name containing "claude"
- **Mistral**: Everything else

### Retry Logic
5 retries with escalating backoff: 5s, 10s, 15s, 20s, 25s

### Statistics Tracked (per faction)
- Total calls, successful calls, failed calls, retried calls
- Success rate
- Tokens in / out / total
- Average tokens per call
- Latency: avg, min, max, p50, p95
- Cost estimate (Haiku pricing)

### JSON Logging
Every game saves a comprehensive JSON log with:
- Benchmark metadata (seed, turns, elapsed, winner, scores)
- Model configuration
- Per-faction LLM statistics
- Aggregate totals + cost estimate
- Final game state (ships, squadrons, BLF status, invasions)
- Full turn log (raw LLM outputs, parsed results, events, commentary)

---

## Complete Action Space

### Faction Actions (Oceania + Eurasia) — up to 3 per week

| Category | Actions |
|----------|---------|
| **Military** | BUILD_SHIP class, BUILD_SQUADRON type, NAVAL_MISSION zone PATROL/ESCORT/BLOCKADE/RAID |
| **Combat** | STRATEGIC_BOMB city, CAS_SUPPORT city, ANTI_SHIP_STRIKE zone, SHORE_BOMBARD zone city, LAY_MINES zone, SWEEP_MINES zone |
| **Economy** | SET_MANUFACTURING 0-1, BUILD_FACTORY type city, REPAIR_FACTORY city, ISSUE_WAR_BONDS city |
| **Budget** | SET_BUDGET cat1 val1 cat2 val2..., ANTI_CORRUPTION |
| **Manpower** | TRAIN_MILITARY city count, CONSCRIPT city count, MOBILIZE_RESERVES city |
| **Intel** | PLANT_SPY city, CODE_BREAK, COUNTER_INTEL, DECEPTION city, CHANGE_CODES |
| **Research** | RESEARCH branch (INDUSTRY/ELECTRONICS/NAVAL/AIR/LAND/DOCTRINE/NUCLEAR/ROCKETRY/CRYPTOGRAPHY/INFRASTRUCTURE) |
| **Invasion** | PLAN_INVASION origin target zone, RECKLESS_INVASION origin target zone, AIRBORNE_INVASION target |
| **Special** | SUPPORT_BLF, DECLARE_WAR_BLF, CONTINUE_SUPPORT_BLF, NOOP |

### Winston Actions — 2 per week

RECRUIT city count, STEAL_ARMS city, SABOTAGE city, BROADCAST, COORDINATE, MOVE city, CONTACT_EURASIA, LIE_LOW, INSPIRE

---

## Running a Game

```bash
# Dry-run (no API calls, random actions)
python tests/benchmark_llm.py --dry-run --turns 100

# Live 100-week game
python tests/benchmark_llm.py \
    --oceania-model claude-haiku-4-5-20251001 \
    --eurasia-model claude-haiku-4-5-20251001 \
    --winston-model claude-haiku-4-5-20251001 \
    --commentary-model mistral-medium-latest \
    --turns 100 \
    --seed 451

# View logs
ls logs/llm_benchmark/
```

### Environment Setup

```bash
# Requires .env with API keys
MISTRAL_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Python 3.14+, dependencies
pip install anthropic mistralai numpy
```

---

## Architecture

```
gravitas/
  llm_game.py          — Main game engine (~2000 lines)
  weather_bridge.py    — Weather system

extensions/
  economy_v2/
    economy_core.py    — GDP, 10 factory types, production
  pop/
    pop_v2.py          — Population (real numbers, 1984 classes)
    pop_state.py       — Legacy population vectors
    pop_params.py      — Archetype parameters
    pop_dynamics.py    — Population ODE dynamics
  research/
    research_system.py — 10×5 tech tree with prerequisites
  governance/
    budget_system.py   — Budget, corruption, bureaucracy
  resistance/
    resistance.py      — BLF 7-level escalation
  naval/
    naval_state.py     — Ship classes, sea zones
    naval_operations.py— Naval combat, missions
    naval_invasion.py  — 3 invasion types
  air_force/
    air_state.py       — Aircraft types, air zones
    air_operations.py  — Air combat, missions
  intelligence/
    intel_system.py    — Fog of war, espionage
  war_economy/
    war_economy_state.py  — Legacy resource model
    manpower.py           — Conscription laws

tests/
  benchmark_llm.py     — Multi-provider LLM benchmark
```
