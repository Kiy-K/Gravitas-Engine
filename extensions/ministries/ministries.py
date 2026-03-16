"""
ministries.py — Autonomous Government Ministries for Air Strip One.

Each faction has 7 ministries that operate semi-autonomously within their
allocated budget. They handle routine tasks, report status to the LLM player,
and request budget increases when needed.

═══════════════════════════════════════════════════════════════════════════════
OCEANIA MINISTRIES (1984 names)

  Ministry of Peace (Minipax)      — Defense: auto-repair ships/planes, rotate
                                     garrisons, reinforce weak sectors
  Ministry of Plenty (Miniplenty)  — Economy: distribute food/fuel, manage
                                     stockpiles, issue emergency rations
  Ministry of Truth (Minitrue)     — Propaganda: manage telescreen network,
                                     boost morale, suppress bad news
  Ministry of Love (Miniluv)       — Security: hunt BLF cells, Thought Police
                                     patrols, counter-intelligence sweeps
  Ministry of Science              — Research: manage tech projects, allocate
                                     researchers, report breakthroughs
  Ministry of Construction         — Infrastructure: repair factories, build
                                     power plants in shortage areas, roads
  Ministry of Labour               — Manpower: training pipeline, conscription
                                     quotas, unemployment management

EURASIA MINISTRIES (Soviet-style names)
  Same 7 divisions with different names and slightly different efficiencies.

═══════════════════════════════════════════════════════════════════════════════
HOW IT WORKS

  1. Player sets budget via SET_BUDGET (allocates GDP across 7 categories)
  2. Each ministry receives its budget share automatically
  3. Ministries spend their budget on ROUTINE tasks every turn:
     - Minipax auto-repairs damaged ships/aircraft below 60% HP
     - Miniplenty auto-distributes food to starving sectors
     - Miniluv auto-patrols for BLF cells in high-risk areas
     - etc.
  4. Ministries report back with:
     - Status summary (what they did this turn)
     - Problems they can't solve alone
     - Budget requests if they need more funding
  5. LLM player sees ministry reports in turn summary and can
     override or redirect priorities with direct orders

  Anti-exploitation:
  - Ministries can only spend their allocated budget (no stealing from others)
  - Efficiency scales with budget: underfunded ministries work poorly
  - Corruption eats a % of every ministry's budget
  - Bureaucracy delays ministry actions (not instant)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════ #
# Ministry Types                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class MinistryType(Enum):
    PEACE          = 0   # Defense/Military — Minipax
    PLENTY         = 1   # Economy/Production — Miniplenty
    TRUTH          = 2   # Propaganda/Media — Minitrue
    LOVE           = 3   # Security/Police — Miniluv
    SCIENCE        = 4   # Research/Technology
    CONSTRUCTION   = 5   # Infrastructure/Building
    LABOUR         = 6   # Manpower/Conscription


# Ministry names per faction
OCEANIA_NAMES = {
    MinistryType.PEACE:        "Ministry of Peace (Minipax)",
    MinistryType.PLENTY:       "Ministry of Plenty (Miniplenty)",
    MinistryType.TRUTH:        "Ministry of Truth (Minitrue)",
    MinistryType.LOVE:         "Ministry of Love (Miniluv)",
    MinistryType.SCIENCE:      "Ministry of Science",
    MinistryType.CONSTRUCTION: "Ministry of Construction",
    MinistryType.LABOUR:       "Ministry of Labour",
}

EURASIA_NAMES = {
    MinistryType.PEACE:        "People's Commissariat of Defense",
    MinistryType.PLENTY:       "State Planning Committee (Gosplan)",
    MinistryType.TRUTH:        "Propaganda Directorate",
    MinistryType.LOVE:         "Committee for State Security",
    MinistryType.SCIENCE:      "Academy of Sciences",
    MinistryType.CONSTRUCTION: "Construction Committee",
    MinistryType.LABOUR:       "Commissariat of Labour",
}

# Maps MinistryType to BudgetCategory index (from governance)
MINISTRY_TO_BUDGET = {
    MinistryType.PEACE:        0,  # MILITARY
    MinistryType.PLENTY:       1,  # PRODUCTION
    MinistryType.SCIENCE:      2,  # RESEARCH
    MinistryType.TRUTH:        3,  # WELFARE (propaganda = public mood management)
    MinistryType.LOVE:         4,  # POLICE
    MinistryType.CONSTRUCTION: 5,  # INFRASTRUCTURE
    MinistryType.LABOUR:       6,  # DEBT_SERVICE (labour = workforce management)
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Ministry Task                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class MinistryTask:
    """A task being processed by a ministry."""
    task_id: int
    description: str
    cost: float                   # GDP cost
    turns_remaining: int = 1      # turns until complete
    turns_total: int = 1
    priority: float = 0.5         # 0-1, higher = more urgent
    auto_generated: bool = True   # was this auto-generated or player-ordered?

    @property
    def progress(self) -> float:
        return 1.0 - (self.turns_remaining / max(self.turns_total, 1))

    @property
    def is_complete(self) -> bool:
        return self.turns_remaining <= 0


# ═══════════════════════════════════════════════════════════════════════════ #
# Ministry                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class Ministry:
    """A single government ministry/department."""
    ministry_type: MinistryType
    faction_id: int
    name: str                             # display name

    # Budget
    budget_allocated: float = 0.0         # GDP received this turn
    budget_spent: float = 0.0             # GDP spent this turn
    budget_requested: float = 0.0         # how much more they want

    # Efficiency (0-1): affected by corruption, staffing, morale
    efficiency: float = 0.8
    staff_morale: float = 0.7             # ministry worker morale
    corruption_loss: float = 0.0          # GDP lost to corruption this turn

    # Tasks
    active_tasks: List[MinistryTask] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)  # this turn's completions

    # Reports back to the leader
    status_report: str = ""
    problems: List[str] = field(default_factory=list)
    budget_request_reason: str = ""

    # Metrics
    total_tasks_completed: int = 0
    total_gdp_spent: float = 0.0
    turns_underfunded: int = 0

    @property
    def is_underfunded(self) -> bool:
        return self.budget_allocated < self.budget_requested * 0.5

    @property
    def workload(self) -> float:
        """How busy is this ministry? 0=idle, 1=overwhelmed."""
        if not self.active_tasks:
            return 0.0
        total_cost = sum(t.cost for t in self.active_tasks)
        return min(1.0, total_cost / max(self.budget_allocated, 1.0))


# ═══════════════════════════════════════════════════════════════════════════ #
# Faction Ministries                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class FactionMinistries:
    """All ministries for one faction."""
    faction_id: int
    ministries: Dict[MinistryType, Ministry] = field(default_factory=dict)

    def get(self, mtype: MinistryType) -> Ministry:
        return self.ministries[mtype]

    def all_reports(self) -> List[str]:
        """Collect all ministry status reports."""
        reports = []
        for m in self.ministries.values():
            if m.status_report:
                reports.append(m.status_report)
        return reports

    def all_problems(self) -> List[str]:
        problems = []
        for m in self.ministries.values():
            problems.extend(m.problems)
        return problems

    def all_budget_requests(self) -> List[str]:
        reqs = []
        for m in self.ministries.values():
            if m.budget_request_reason:
                reqs.append(f"  {m.name}: {m.budget_request_reason}")
        return reqs


@dataclass
class MinistryWorld:
    """Ministry state for all factions."""
    factions: Dict[int, FactionMinistries] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def initialize_ministries(faction_ids: List[int]) -> MinistryWorld:
    """Create ministries for all factions."""
    world = MinistryWorld()
    for fid in faction_ids:
        names = OCEANIA_NAMES if fid == 0 else EURASIA_NAMES
        fm = FactionMinistries(faction_id=fid)
        for mtype in MinistryType:
            # Base efficiency varies by government type
            if fid == 0:  # Oceania — totalitarian: high security, low innovation
                eff_map = {
                    MinistryType.PEACE: 0.85, MinistryType.PLENTY: 0.65,
                    MinistryType.TRUTH: 0.90, MinistryType.LOVE: 0.95,
                    MinistryType.SCIENCE: 0.60, MinistryType.CONSTRUCTION: 0.70,
                    MinistryType.LABOUR: 0.80,
                }
            else:  # Eurasia — communist: committee layers slow everything
                eff_map = {
                    MinistryType.PEACE: 0.80, MinistryType.PLENTY: 0.60,
                    MinistryType.TRUTH: 0.75, MinistryType.LOVE: 0.85,
                    MinistryType.SCIENCE: 0.65, MinistryType.CONSTRUCTION: 0.65,
                    MinistryType.LABOUR: 0.85,
                }
            fm.ministries[mtype] = Ministry(
                ministry_type=mtype,
                faction_id=fid,
                name=names[mtype],
                efficiency=eff_map.get(mtype, 0.7),
            )
        world.factions[fid] = fm
    return world


# ═══════════════════════════════════════════════════════════════════════════ #
# Ministry Step — each ministry performs autonomous tasks                     #
# ═══════════════════════════════════════════════════════════════════════════ #

_next_task_id = 0

def _gen_task_id() -> int:
    global _next_task_id
    _next_task_id += 1
    return _next_task_id


def step_ministries(
    world: MinistryWorld,
    game: Any,  # GameState — import avoided to prevent circular
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[MinistryWorld, Dict[str, Any]]:
    """Step all ministries. Each ministry performs autonomous tasks."""
    feedback: Dict[str, Any] = {}

    for fid, fm in world.factions.items():
        faction_fb = {}

        # Get budget allocation from governance (if available)
        budget_alloc = np.zeros(7)
        gdp = 100.0
        corruption_rate = 0.08
        if game.governance is not None and fid in game.governance.factions:
            gb = game.governance.factions[fid]
            budget_alloc = gb.allocation.copy()
            gdp = gb.gdp_revenue
            corruption_rate = gb.corruption.effective_rate

        for mtype, ministry in fm.ministries.items():
            # Reset turn state
            ministry.completed_tasks = []
            ministry.problems = []
            ministry.budget_request_reason = ""
            ministry.budget_spent = 0.0
            ministry.corruption_loss = 0.0

            # Allocate budget from governance categories
            budget_idx = MINISTRY_TO_BUDGET.get(mtype, 0)
            raw_budget = budget_alloc[budget_idx] * gdp
            corruption_cost = raw_budget * corruption_rate
            ministry.corruption_loss = corruption_cost
            ministry.budget_allocated = raw_budget - corruption_cost

            # Efficiency decays if underfunded
            if ministry.budget_allocated < 5.0:
                ministry.efficiency = max(0.3, ministry.efficiency - 0.02 * dt)
                ministry.turns_underfunded += 1
            else:
                ministry.efficiency = min(0.95, ministry.efficiency + 0.01 * dt)
                ministry.turns_underfunded = 0

            # ── Process existing tasks ──────────────────────────────── #
            remaining = []
            for task in ministry.active_tasks:
                task.turns_remaining -= 1
                ministry.budget_spent += task.cost / max(task.turns_total, 1)
                if task.is_complete:
                    ministry.completed_tasks.append(task.description)
                    ministry.total_tasks_completed += 1
                else:
                    remaining.append(task)
            ministry.active_tasks = remaining

            # ── Generate auto-tasks based on ministry type ──────────── #
            _generate_auto_tasks(ministry, game, fid, rng, dt)

            # ── Generate status report ──────────────────────────────── #
            _generate_report(ministry, game, fid)

            ministry.total_gdp_spent += ministry.budget_spent

        feedback[fid] = faction_fb

    return world, feedback


def _generate_auto_tasks(ministry: Ministry, game: Any, fid: int, rng: np.random.Generator, dt: float):
    """Generate autonomous tasks for each ministry type."""
    budget = ministry.budget_allocated
    eff = ministry.efficiency

    # Don't generate tasks if we're already overwhelmed
    if len(ministry.active_tasks) >= 3:
        return

    if ministry.ministry_type == MinistryType.PEACE:
        # ── Minipax: auto-repair damaged ships, rotate tired squadrons ── #
        if game.naval is not None:
            damaged = [s for s in game.naval.faction_ships(fid) if s.hp < s.max_hp * 0.6]
            if damaged and budget > 3.0:
                ship = damaged[0]
                repair_amt = min(budget * 0.1 * eff, ship.max_hp * 0.15)
                ship.hp = min(ship.max_hp, ship.hp + repair_amt)
                ministry.active_tasks.append(MinistryTask(
                    _gen_task_id(), f"Repairing {ship.stats.ship_class.name} (hull integrity +{repair_amt:.0f})",
                    cost=repair_amt * 0.5, turns_remaining=1, turns_total=1,
                ))
            if not damaged and budget > 5.0:
                ministry.budget_request_reason = ""  # fleet in good shape
            elif len(damaged) > 3:
                ministry.budget_request_reason = f"URGENT: {len(damaged)} ships need repair. Need more funding."

        # Auto-repair damaged air squadrons
        if game.air is not None:
            weak_sqs = [sq for sq in game.air.faction_squadrons(fid) if sq.strength < 0.5]
            if weak_sqs and budget > 2.0:
                sq = weak_sqs[0]
                sq.strength = min(1.0, sq.strength + 0.1 * eff)
                sq.fuel = min(1.0, sq.fuel + 0.2)
                ministry.completed_tasks.append(f"Refitted squadron (strength +10%)")

    elif ministry.ministry_type == MinistryType.PLENTY:
        # ── Miniplenty: auto-distribute food to starving sectors ── #
        if game.war_economy is not None:
            from extensions.war_economy.war_economy_state import Resource
            starving = []
            surplus = []
            for ce in game.war_economy.cluster_economies:
                if game.cluster_owners.get(ce.cluster_id) != fid:
                    continue
                food_ratio = ce.stockpile_ratio(Resource.PROCESSED_FOOD)
                if food_ratio < 0.15:
                    starving.append(ce)
                elif food_ratio > 0.5:
                    surplus.append(ce)

            # Auto-redistribute from surplus to starving
            if starving and surplus and budget > 2.0:
                for s_ce in starving[:2]:
                    s_ce.resource_stockpile[Resource.PROCESSED_FOOD.value] += 5.0 * eff
                    ministry.completed_tasks.append(
                        f"Emergency rations to {game.cluster_names[s_ce.cluster_id]}")

            if len(starving) > 3:
                ministry.budget_request_reason = f"FOOD CRISIS: {len(starving)} sectors starving!"
                ministry.problems.append(f"⚠ {len(starving)} sectors below 15% food — increase WELFARE spending!")

    elif ministry.ministry_type == MinistryType.TRUTH:
        # ── Minitrue: auto-manage propaganda, boost morale ── #
        if budget > 3.0:
            # Propaganda effect: small trust boost in own sectors
            boost = 0.002 * eff * dt
            for cid, owner in game.cluster_owners.items():
                if owner == fid and cid < len(game.cluster_data):
                    game.cluster_data[cid, 4] = min(1.0, game.cluster_data[cid, 4] + boost)
                    game.cluster_data[cid, 5] = max(0.0, game.cluster_data[cid, 5] - boost * 0.5)
            ministry.completed_tasks.append("Telescreen broadcasts: Victory is inevitable. Morale maintained.")

        # Check for low-trust sectors
        low_trust = []
        for cid, owner in game.cluster_owners.items():
            if owner == fid and cid < len(game.cluster_data):
                if game.cluster_data[cid, 4] < 0.3:
                    low_trust.append(game.cluster_names[cid] if cid < len(game.cluster_names) else f"C{cid}")
        if low_trust:
            ministry.problems.append(f"Low trust in: {', '.join(low_trust[:3])}")

    elif ministry.ministry_type == MinistryType.LOVE:
        # ── Miniluv: auto-sweep for BLF, manage Thought Police ── #
        if game.resistance is not None and budget > 2.0:
            blf = game.resistance
            # Auto Thought Police patrols in high-risk areas
            if blf.escalation.value >= 2:
                for cell in blf.active_cells[:2]:
                    # Small chance of detecting a cell
                    detect_chance = 0.05 * eff * (budget / 10.0)
                    if rng.random() < detect_chance:
                        cell.detection_risk = min(1.0, cell.detection_risk + 0.1)
                        ministry.completed_tasks.append(
                            f"Thought Police narrowing search in {game.cluster_names[cell.cluster_id] if cell.cluster_id < len(game.cluster_names) else 'sector'}")

            if blf.escalation.value >= 3:
                ministry.budget_request_reason = "ORGANIZED RESISTANCE detected. Need more agents."
                ministry.problems.append(f"⚠ BLF at level {blf.escalation.value} — increase POLICE budget!")

    elif ministry.ministry_type == MinistryType.SCIENCE:
        # ── Ministry of Science: report research status ── #
        if game.research is not None:
            fr = game.research.factions.get(fid)
            if fr:
                active = [p for p in fr.active_projects if not p.is_complete]
                free_slots = fr.max_simultaneous - len(active)
                if free_slots > 0:
                    ministry.problems.append(f"⚠ {free_slots} research slot(s) IDLE — order RESEARCH to start a project!")
                if active:
                    from extensions.research.research_system import TECH_TREE
                    for p in active:
                        bonus = TECH_TREE[p.branch][p.tier]
                        turns_left = p.required_turns - int(p.progress_pct * p.required_turns)
                        ministry.completed_tasks.append(
                            f"Research: {bonus.name} ({p.progress_pct:.0%}, ~{turns_left} weeks)")

                if budget < 3.0:
                    ministry.budget_request_reason = "Research STALLED — funding below minimum threshold."

    elif ministry.ministry_type == MinistryType.CONSTRUCTION:
        # ── Construction: auto-repair damaged factories ── #
        if game.economy is not None and budget > 3.0:
            for ce in game.economy.clusters:
                if game.cluster_owners.get(ce.cluster_id) != fid:
                    continue
                # Auto-repair damaged factories
                for f in ce.factories:
                    if hasattr(f, 'damage') and f.damage > 0.2 and not f.is_building:
                        f.damage = max(0.0, f.damage - 0.05 * eff)
                        ministry.completed_tasks.append(
                            f"Repaired {f.factory_type.name} in {game.cluster_names[ce.cluster_id] if ce.cluster_id < len(game.cluster_names) else 'sector'}")
                        break  # one repair per turn

            # Check for power shortages
            power_short = []
            for ce in game.economy.clusters:
                if game.cluster_owners.get(ce.cluster_id) == fid and ce.power_ratio < 0.6:
                    power_short.append(game.cluster_names[ce.cluster_id] if ce.cluster_id < len(game.cluster_names) else f"C{ce.cluster_id}")
            if power_short:
                ministry.problems.append(f"POWER SHORTAGE in: {', '.join(power_short[:3])}")
                if len(power_short) > 3:
                    ministry.budget_request_reason = f"CRITICAL: {len(power_short)} sectors without power!"

    elif ministry.ministry_type == MinistryType.LABOUR:
        # ── Labour: report manpower status, auto-manage training ── #
        total_unemp = 0
        total_training = 0
        for i, mp in enumerate(game.manpower_clusters):
            if game.cluster_owners.get(i) != fid:
                continue
            total_unemp += mp.unemployed
            total_training += mp.total_in_training

        if total_unemp > 5000:
            ministry.problems.append(f"HIGH UNEMPLOYMENT: {total_unemp:.0f}k idle workers")
        if total_training > 0:
            ministry.completed_tasks.append(f"Training pipeline: {total_training} recruits in progress")
        if budget < 2.0:
            ministry.budget_request_reason = "Training budget insufficient. Conscripts arriving untrained."


def _generate_report(ministry: Ministry, game: Any, fid: int):
    """Generate the ministry's status report for the LLM."""
    parts = []
    short_name = ministry.name.split("(")[0].strip() if "(" in ministry.name else ministry.name

    # Budget status
    budget_status = "adequate" if ministry.budget_allocated > 5.0 else "LOW" if ministry.budget_allocated > 2.0 else "CRITICAL"
    parts.append(f"[{short_name}] Budget: {ministry.budget_allocated:.0f} GDP ({budget_status})")

    # Completed tasks
    if ministry.completed_tasks:
        for task in ministry.completed_tasks[:2]:
            parts.append(f"  ✓ {task}")

    # Problems
    if ministry.problems:
        for prob in ministry.problems[:2]:
            parts.append(f"  {prob}")

    # Budget request
    if ministry.budget_request_reason:
        parts.append(f"  💰 REQUEST: {ministry.budget_request_reason}")

    ministry.status_report = "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════ #
# Reports for turn summary                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def ministry_reports(world: MinistryWorld, faction_id: int) -> str:
    """Generate condensed ministry reports for LLM turn summary."""
    fm = world.factions.get(faction_id)
    if fm is None:
        return ""

    lines = []
    for mtype in MinistryType:
        m = fm.ministries.get(mtype)
        if m is None:
            continue
        # One-line summary per ministry
        short = m.name.split("(")[0].strip() if "(" in m.name else m.name
        status_parts = []

        # Budget
        if m.budget_allocated < 2.0:
            status_parts.append("UNDERFUNDED")
        elif m.budget_allocated < 5.0:
            status_parts.append("low budget")

        # Tasks done
        if m.completed_tasks:
            status_parts.append(f"{len(m.completed_tasks)} tasks done")

        # Problems
        if m.problems:
            status_parts.append(f"{len(m.problems)} issues")

        # Budget request
        if m.budget_request_reason:
            status_parts.append("REQUESTING MORE FUNDS")

        if status_parts:
            lines.append(f"  {short}: {', '.join(status_parts)}")
        else:
            lines.append(f"  {short}: operating normally")

    # Urgent problems across all ministries
    all_problems = fm.all_problems()
    if all_problems:
        lines.append("  URGENT ISSUES:")
        for p in all_problems[:3]:
            lines.append(f"    {p}")

    # Budget requests
    all_requests = fm.all_budget_requests()
    if all_requests:
        lines.append("  BUDGET REQUESTS:")
        for r in all_requests[:3]:
            lines.append(f"    {r}")

    return "\n".join(lines)
