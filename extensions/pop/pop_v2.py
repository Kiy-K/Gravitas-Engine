"""
pop_v2.py — Realistic population system with real numbers, 1984 social classes,
jobs, political power, and conscription scaling.

═══════════════════════════════════════════════════════════════════════════════
DESIGN: REAL NUMBERS, NOT [0,1]

  London has 10,000,000 people. Not 0.85. TEN MILLION.
  Birmingham has 3,500,000. Glasgow has 2,800,000. Dover has 120,000.
  This matters because:
    - Conscripting 100,000 from London (1%) is different from Dover (83%)
    - GDP scales with real workforce
    - Factory output depends on real worker count
    - Military strength is absolute, not relative

═══════════════════════════════════════════════════════════════════════════════
1984 SOCIAL CLASSES (from the novel)

  INNER PARTY (2%): The ruling elite. O'Brien's class. They run the Ministries,
    control the Thought Police, live in luxury. Politically omnipotent.
    In Eurasia: Politburo + Central Committee equivalents.

  OUTER PARTY (13%): Winston's former class. Bureaucrats, administrators,
    technicians. Constantly surveilled. The ones who actually run the machine.
    Most susceptible to thoughtcrime. Key source of skilled workers.

  PROLES (85%): The masses. Unsurveilled, uneducated, politically inert —
    UNTIL they're not. The Party says "proles and animals are free."
    Key source of factory workers, soldiers, and revolutionary potential.

═══════════════════════════════════════════════════════════════════════════════
JOB CATEGORIES (vectorized per class)

  Each class has a job distribution vector J[8]:
    0  FARMER          — food production (mostly proles)
    1  FACTORY_WORKER  — industry (mostly proles + outer party technicians)
    2  MINER           — coal, ore extraction (proles)
    3  SOLDIER         — active military (all classes, mostly proles)
    4  BUREAUCRAT      — administration (outer + inner party)
    5  TECHNICIAN      — skilled labor, engineering (outer party)
    6  POLICE          — Thought Police + regular police (outer + inner)
    7  UNEMPLOYED      — available for conscription or factory work

═══════════════════════════════════════════════════════════════════════════════
POLITICAL POWER

  Each class has political_power per capita:
    Inner Party: 50.0 (one Inner Party member = 50 proles politically)
    Outer Party:  3.0 (some influence through institutions)
    Proles:       0.2 (almost none, but 85% of the population)

  Total political power = Σ(class_population × class_power_per_capita)
  This determines: regime stability, policy options, conscription limits

═══════════════════════════════════════════════════════════════════════════════
CONSCRIPTION SCALING

  Conscription Law     | % of working-age that can be drafted
  VOLUNTEER_ONLY       | 2%  (standing army volunteers only)
  LIMITED_DRAFT        | 5%
  GENERAL_MOBILISATION | 12%
  TOTAL_MOBILISATION   | 25%
  SCRAPING_THE_BARREL  | 40% (economic collapse territory)

  Real numbers: London (10M, 6M working age, 25% = 1.5M soldiers possible)
  vs Dover (120K, 72K working age, 25% = 18K soldiers possible)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════ #
# Social Classes                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class SocialClass(Enum):
    INNER_PARTY = 0   # 2% — ruling elite
    OUTER_PARTY = 1   # 13% — bureaucrats, technicians
    PROLES      = 2   # 85% — the masses

# Default class distribution
CLASS_SHARES = {
    SocialClass.INNER_PARTY: 0.02,
    SocialClass.OUTER_PARTY: 0.13,
    SocialClass.PROLES:      0.85,
}

# Political power per capita
POLITICAL_POWER_PER_CAPITA = {
    SocialClass.INNER_PARTY: 50.0,
    SocialClass.OUTER_PARTY:  3.0,
    SocialClass.PROLES:       0.2,
}

# Income multiplier per class
CLASS_INCOME = {
    SocialClass.INNER_PARTY: 5.0,   # luxury goods, special shops
    SocialClass.OUTER_PARTY: 1.2,   # Victory Gin and Victory Cigarettes
    SocialClass.PROLES:      0.4,   # subsistence
}

N_CLASSES = 3
N_JOBS = 8


# ═══════════════════════════════════════════════════════════════════════════ #
# Job Categories                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class JobType(Enum):
    FARMER          = 0
    FACTORY_WORKER  = 1
    MINER           = 2
    SOLDIER         = 3
    BUREAUCRAT      = 4
    TECHNICIAN      = 5
    POLICE          = 6
    UNEMPLOYED      = 7


# ═══════════════════════════════════════════════════════════════════════════ #
# Cluster Population                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class ClusterPop:
    """Population state for one cluster — REAL NUMBERS."""
    cluster_id: int

    # Total population (real number)
    total_pop: int = 1_000_000

    # Class populations (real numbers, sum = total_pop)
    inner_party: int = 20_000      # 2%
    outer_party: int = 130_000     # 13%
    proles: int = 850_000          # 85%

    # Job distribution matrix: jobs[class_idx][job_idx] = count
    # Shape: (3, 8) — 3 classes × 8 jobs
    # Stored as flat array for vectorization
    jobs: np.ndarray = field(default_factory=lambda: np.zeros((N_CLASSES, N_JOBS), dtype=np.int64))

    # Demographics
    working_age_ratio: float = 0.62     # 62% are 16-65
    # Class-based birth rates (per person per WEEK)
    # Proles: ~4-5 children per family, large households, uneducated
    # Outer Party: ~1-2 children, small families, surveilled
    # Inner Party: ~0-1 children, almost no families, married to the Party
    birth_rate_proles: float = 0.00045  # ~2.3% annual — large prole families
    birth_rate_outer: float = 0.00015   # ~0.8% annual — small families
    birth_rate_inner: float = 0.00004   # ~0.2% annual — almost no reproduction
    death_rate: float = 0.00020         # ~1.0% annual natural death rate
    war_deaths_this_turn: int = 0
    civilian_deaths_this_turn: int = 0
    births_this_turn: int = 0
    starvation_deaths_this_turn: int = 0
    bombing_deaths_this_turn: int = 0
    new_manpower_this_turn: int = 0     # grown-up pop entering workforce

    # Military state
    active_military: int = 0
    reserves: int = 0
    in_training: int = 0
    training_turns_left: int = 0
    wounded: int = 0
    pow_held: int = 0                   # prisoners we hold
    pow_lost: int = 0                   # our soldiers captured by enemy

    # Morale & satisfaction (per class, 3-element vector)
    satisfaction: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.5, 0.4], dtype=np.float64))
    radicalization: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.1, 0.2], dtype=np.float64))

    # Conscription
    conscription_exhaustion: float = 0.0  # 0-1, how tapped out the population is

    @property
    def working_age(self) -> int:
        return int(self.total_pop * self.working_age_ratio)

    @property
    def total_military(self) -> int:
        return self.active_military + self.reserves + self.in_training

    @property
    def military_ratio(self) -> float:
        wa = self.working_age
        return self.total_military / max(wa, 1)

    @property
    def total_employed(self) -> int:
        return int(self.jobs.sum()) - int(self.jobs[:, JobType.UNEMPLOYED.value].sum())

    @property
    def total_unemployed(self) -> int:
        return int(self.jobs[:, JobType.UNEMPLOYED.value].sum())

    @property
    def unemployment_rate(self) -> float:
        wa = self.working_age
        return self.total_unemployed / max(wa, 1)

    @property
    def political_power(self) -> float:
        """Total political power in this cluster."""
        return (self.inner_party * POLITICAL_POWER_PER_CAPITA[SocialClass.INNER_PARTY] +
                self.outer_party * POLITICAL_POWER_PER_CAPITA[SocialClass.OUTER_PARTY] +
                self.proles * POLITICAL_POWER_PER_CAPITA[SocialClass.PROLES])

    @property
    def prole_fraction(self) -> float:
        return self.proles / max(self.total_pop, 1)

    @property
    def class_vector(self) -> np.ndarray:
        """[inner_party, outer_party, proles] as array."""
        return np.array([self.inner_party, self.outer_party, self.proles], dtype=np.int64)

    @property
    def available_for_conscription(self) -> int:
        """Working-age unemployed proles + outer party (inner party exempt)."""
        prole_unemployed = int(self.jobs[SocialClass.PROLES.value, JobType.UNEMPLOYED.value])
        outer_unemployed = int(self.jobs[SocialClass.OUTER_PARTY.value, JobType.UNEMPLOYED.value])
        return prole_unemployed + outer_unemployed

    @property
    def available_for_work(self) -> int:
        """Total unemployed across all classes."""
        return self.total_unemployed

    @property
    def factory_workers(self) -> int:
        return int(self.jobs[:, JobType.FACTORY_WORKER.value].sum())

    @property
    def farmers(self) -> int:
        return int(self.jobs[:, JobType.FARMER.value].sum())

    @property
    def miners(self) -> int:
        return int(self.jobs[:, JobType.MINER.value].sum())

    @property
    def police_count(self) -> int:
        return int(self.jobs[:, JobType.POLICE.value].sum())


# ═══════════════════════════════════════════════════════════════════════════ #
# Population World                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class PopWorld:
    """Population state for all clusters."""
    clusters: List[ClusterPop] = field(default_factory=list)
    turn: int = 0


# ═══════════════════════════════════════════════════════════════════════════ #
# Real Population Data for 32-Sector Map (1984 setting)                        #
# ═══════════════════════════════════════════════════════════════════════════ #

# Realistic population in thousands (1984 Orwellian Britain + France)
# 1958 census-era population data (thousands) — closest to the 1984 timeline
CLUSTER_POPULATIONS = {
    # Oceania — British Isles (1951/1961 UK Census interpolated to ~1958)
    0:   7_500,  # London — Greater London (peaked ~8.6M in 1939, declining by 1958)
    1:      25,  # Dover — small Channel fortress town
    2:     247,  # Portsmouth — naval city (1951 census)
    3:     180,  # Southampton — port + industry
    4:      30,  # Canterbury — cathedral + garrison town
    5:     150,  # Brighton — south coast resort + garrison
    6:     420,  # Bristol — western hub (1951-1961 avg)
    7:     113,  # Plymouth — naval base (1951 census)
    8:     250,  # Cardiff — Welsh coal + steel capital
    9:     970,  # Birmingham — second city (1951: 922K, 1961: 1.01M)
    10:    640,  # Manchester — northern powerhouse (1951: 730K declining)
    11:    700,  # Liverpool — Atlantic port (1951: 803K declining)
    12:    480,  # Leeds — northern industry + rail (1951: 458K)
    13:    130,  # Norwich — East Anglia agricultural hub
    14:    470,  # Edinburgh — Scottish capital
    15:    900,  # Glasgow — Clyde shipyards (peak industrial era)
    16:    650,  # Dublin — Airstrip Two capital (1956 Irish census)
    17:    440,  # Belfast — Harland & Wolff shipyards
    # Eurasia — France + Benelux (1954 French Census / 1958 Belgian)
    18:     75,  # Calais — Channel port, garrison
    19:     25,  # Dunkirk — fleet base (post-war rebuilding)
    20:    140,  # Le Havre — Normandy port (rebuilt after D-Day destruction)
    21:     35,  # Cherbourg — submarine base
    22:    110,  # Amiens — Picardy rail junction
    23:    115,  # Rouen — Seine industrial city, steel mills
    24:    195,  # Lille — Nord-Pas-de-Calais industrial + coal
    25:  1_100,  # Brussels — Belgian capital, Benelux Command
    26:    260,  # Antwerp — major port, North Sea fleet
    27:    750,  # Rotterdam — Europoort, North Sea Fleet Pride
    28:    870,  # Amsterdam — trade, finance, industry
    29:    200,  # Luxembourg — steel industry, quiet rear area
    30:  5_000,  # Paris — metro area (1954 census: ~5M metro)
    31:     85,  # Orleans — Loire logistics hub
    32:    530,  # Lyon — Rhône-Alpes industrial center
    33:    120,  # Brest — Finistère naval base, submarine pens
    34:    250,  # Bordeaux — Gironde, southern reserves
}


# ═══════════════════════════════════════════════════════════════════════════ #
# Initialization                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def _init_jobs(total_pop: int, inner: int, outer: int, proles: int,
               role: str, rng: np.random.Generator) -> np.ndarray:
    """Initialize job distribution matrix based on cluster role."""
    jobs = np.zeros((N_CLASSES, N_JOBS), dtype=np.int64)

    # Inner Party: mostly bureaucrats + police
    ip_wa = int(inner * 0.62)
    jobs[0, JobType.BUREAUCRAT.value] = int(ip_wa * 0.5)
    jobs[0, JobType.POLICE.value] = int(ip_wa * 0.2)
    jobs[0, JobType.TECHNICIAN.value] = int(ip_wa * 0.1)
    jobs[0, JobType.UNEMPLOYED.value] = max(0, ip_wa - int(jobs[0].sum()))

    # Outer Party: bureaucrats + technicians + some factory/police
    op_wa = int(outer * 0.62)
    jobs[1, JobType.BUREAUCRAT.value] = int(op_wa * 0.3)
    jobs[1, JobType.TECHNICIAN.value] = int(op_wa * 0.25)
    jobs[1, JobType.FACTORY_WORKER.value] = int(op_wa * 0.15)
    jobs[1, JobType.POLICE.value] = int(op_wa * 0.1)
    jobs[1, JobType.SOLDIER.value] = int(op_wa * 0.05)
    jobs[1, JobType.UNEMPLOYED.value] = max(0, op_wa - int(jobs[1].sum()))

    # Proles: role-dependent
    pr_wa = int(proles * 0.62)
    if role in ("industrial", "capital"):
        jobs[2, JobType.FACTORY_WORKER.value] = int(pr_wa * 0.40)
        jobs[2, JobType.FARMER.value] = int(pr_wa * 0.05)
        jobs[2, JobType.MINER.value] = int(pr_wa * 0.05)
    elif role in ("agricultural",):
        jobs[2, JobType.FARMER.value] = int(pr_wa * 0.50)
        jobs[2, JobType.FACTORY_WORKER.value] = int(pr_wa * 0.10)
    elif role in ("naval_base", "port_city", "sub_base"):
        jobs[2, JobType.FACTORY_WORKER.value] = int(pr_wa * 0.35)
        jobs[2, JobType.FARMER.value] = int(pr_wa * 0.05)
        jobs[2, JobType.MINER.value] = int(pr_wa * 0.02)
    elif role in ("garrison",):
        jobs[2, JobType.SOLDIER.value] = int(pr_wa * 0.15)
        jobs[2, JobType.FACTORY_WORKER.value] = int(pr_wa * 0.20)
        jobs[2, JobType.FARMER.value] = int(pr_wa * 0.10)
    else:
        jobs[2, JobType.FACTORY_WORKER.value] = int(pr_wa * 0.30)
        jobs[2, JobType.FARMER.value] = int(pr_wa * 0.10)

    # Fill remaining as unemployed
    for c in range(N_CLASSES):
        assigned = int(jobs[c].sum())
        wa = [ip_wa, op_wa, pr_wa][c]
        jobs[c, JobType.UNEMPLOYED.value] = max(0, wa - assigned)

    return jobs


def initialize_pop_v2(
    n_clusters: int,
    cluster_owners: Dict[int, int],
    rng: np.random.Generator,
) -> PopWorld:
    """Initialize realistic population for all 32 clusters."""
    from extensions.economy_v2.economy_core import CLUSTER_ROLES

    clusters = []
    for cid in range(n_clusters):
        pop_k = CLUSTER_POPULATIONS.get(cid, 500)
        total = pop_k * 1000  # convert from thousands

        inner = int(total * CLASS_SHARES[SocialClass.INNER_PARTY])
        outer = int(total * CLASS_SHARES[SocialClass.OUTER_PARTY])
        proles = total - inner - outer

        role = CLUSTER_ROLES.get(cid, "default")
        jobs = _init_jobs(total, inner, outer, proles, role, rng)

        # Military: frontline clusters have more soldiers
        mil_ratio = 0.03  # 3% baseline
        if cid in (1, 5, 18, 19, 20, 21):  # frontline
            mil_ratio = 0.08
        elif cid in (2, 7, 14, 17, 26):  # naval bases
            mil_ratio = 0.05

        wa = int(total * 0.62)
        active_mil = int(wa * mil_ratio * 0.6)
        reserves = int(wa * mil_ratio * 0.25)
        garrison = int(wa * mil_ratio * 0.15)

        # Satisfaction: Inner Party happy, Outer Party nervous, Proles resigned
        owner = cluster_owners.get(cid, 0)
        if owner == 0:  # Oceania
            sat = np.array([0.80, 0.45, 0.35], dtype=np.float64)
            rad = np.array([0.02, 0.08, 0.15], dtype=np.float64)
        else:  # Eurasia
            sat = np.array([0.75, 0.50, 0.40], dtype=np.float64)
            rad = np.array([0.03, 0.06, 0.12], dtype=np.float64)

        cp = ClusterPop(
            cluster_id=cid,
            total_pop=total,
            inner_party=inner,
            outer_party=outer,
            proles=proles,
            jobs=jobs,
            active_military=active_mil,
            reserves=reserves,
            in_training=0,
            satisfaction=sat,
            radicalization=rad,
        )
        clusters.append(cp)

    return PopWorld(clusters=clusters)


# ═══════════════════════════════════════════════════════════════════════════ #
# Population Step                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def step_pop_v2(
    world: PopWorld,
    cluster_owners: Dict[int, int],
    food_available: Dict[int, float],  # per cluster, from Economy V2
    rng: np.random.Generator,
    dt: float = 1.0,
) -> Tuple[PopWorld, Dict[str, Any]]:
    """Advance population by one turn."""
    feedback: Dict[str, Any] = {}

    total_births_all = 0
    total_deaths_all = 0

    for cp in world.clusters:
        cp.births_this_turn = 0
        cp.starvation_deaths_this_turn = 0
        cp.bombing_deaths_this_turn = 0
        cp.new_manpower_this_turn = 0

        # ── 1. Class-based population growth (Proles breed fast!) ───── #
        # Proles: large families, uneducated, 4-5 children per woman
        # Outer Party: small families, surveilled, stressed, 1-2 children
        # Inner Party: married to the Party, almost zero reproduction
        prole_births = int(cp.proles * cp.birth_rate_proles * dt)
        outer_births = int(cp.outer_party * cp.birth_rate_outer * dt)
        inner_births = int(cp.inner_party * cp.birth_rate_inner * dt)

        # Starvation suppresses birth rate
        food = food_available.get(cp.cluster_id, 50.0)
        food_per_capita = food / max(cp.total_pop / 100_000, 0.01)
        if food_per_capita < 0.3:
            prole_births = int(prole_births * food_per_capita / 0.3)

        # Add births to population
        cp.proles += prole_births
        cp.outer_party += outer_births
        cp.inner_party += inner_births
        total_births = prole_births + outer_births + inner_births
        cp.births_this_turn = total_births
        total_births_all += total_births

        # ── 1b. Natural deaths (age, disease) ─────────────────────── #
        natural_deaths_p = int(cp.proles * cp.death_rate * dt)
        natural_deaths_o = int(cp.outer_party * cp.death_rate * 0.8 * dt)  # better healthcare
        natural_deaths_i = int(cp.inner_party * cp.death_rate * 0.5 * dt)  # best healthcare
        cp.proles = max(100, cp.proles - natural_deaths_p)
        cp.outer_party = max(10, cp.outer_party - natural_deaths_o)
        cp.inner_party = max(1, cp.inner_party - natural_deaths_i)
        total_natural = natural_deaths_p + natural_deaths_o + natural_deaths_i
        total_deaths_all += total_natural

        cp.total_pop = cp.inner_party + cp.outer_party + cp.proles

        # ── 1c. Grown-up pop enters manpower pool ─────────────────── #
        # Each week, a fraction of the population "comes of age" (turns 16)
        # and enters the working-age pool as unemployed proles/outer party
        new_workers = int(cp.total_pop * 0.00025 * dt)  # ~1.3% annual maturation
        # Most new workers are proles (85% of births were proles)
        new_prole_workers = int(new_workers * 0.88)
        new_outer_workers = new_workers - new_prole_workers
        cp.jobs[SocialClass.PROLES.value, JobType.UNEMPLOYED.value] += new_prole_workers
        cp.jobs[SocialClass.OUTER_PARTY.value, JobType.UNEMPLOYED.value] += new_outer_workers
        cp.new_manpower_this_turn = new_workers

        # ── 2. Food → starvation (Proles die first, Inner Party last) ─── #
        if food_per_capita < 0.3:
            severity = (0.3 - food_per_capita) * 10  # 0-3 scale
            # Proles starve first (no access to Party stores)
            starve_proles = int(cp.proles * 0.002 * severity * dt)
            # Outer Party: some access to rations
            starve_outer = int(cp.outer_party * 0.0005 * severity * dt) if food_per_capita < 0.15 else 0
            # Inner Party: always fed (special stores), only starve in total famine
            starve_inner = int(cp.inner_party * 0.0001 * severity * dt) if food_per_capita < 0.05 else 0

            cp.proles = max(100, cp.proles - starve_proles)
            cp.outer_party = max(10, cp.outer_party - starve_outer)
            cp.inner_party = max(1, cp.inner_party - starve_inner)
            total_starve = starve_proles + starve_outer + starve_inner
            cp.starvation_deaths_this_turn = total_starve
            cp.civilian_deaths_this_turn += total_starve
            total_deaths_all += total_starve
            cp.total_pop = cp.inner_party + cp.outer_party + cp.proles

            # Starvation radicalizes proles
            cp.radicalization[SocialClass.PROLES.value] = min(1.0,
                cp.radicalization[SocialClass.PROLES.value] + 0.02 * severity * dt)
            cp.satisfaction[SocialClass.PROLES.value] = max(0.0,
                cp.satisfaction[SocialClass.PROLES.value] - 0.03 * severity * dt)

        # ── 2b. Bombing casualties (from cluster hazard level) ───────── #
        if cp.cluster_id < len(world.clusters):
            # Higher hazard = more civilian deaths from bombing
            # hazard is in cluster_data[cid, 1] but we approximate from satisfaction
            hazard_proxy = max(0, 1.0 - float(cp.satisfaction.mean()))
            if hazard_proxy > 0.3:
                bomb_deaths = int(cp.proles * 0.0005 * (hazard_proxy - 0.3) * dt)
                cp.proles = max(100, cp.proles - bomb_deaths)
                cp.bombing_deaths_this_turn = bomb_deaths
                cp.civilian_deaths_this_turn += bomb_deaths
                total_deaths_all += bomb_deaths
                cp.total_pop = cp.inner_party + cp.outer_party + cp.proles

        # ── 3. Training pipeline ──────────────────────────────────────── #
        if cp.in_training > 0 and cp.training_turns_left > 0:
            cp.training_turns_left -= 1
            if cp.training_turns_left <= 0:
                cp.active_military += cp.in_training
                cp.in_training = 0

        # ── 4. Wounded → healed (slow) ────────────────────────────────── #
        if cp.wounded > 0:
            heal = max(1, int(cp.wounded * 0.05 * dt))
            cp.wounded -= heal
            cp.reserves += heal

        # ── 5. Satisfaction / radicalization dynamics ─────────────────── #
        for c in range(N_CLASSES):
            # Satisfaction decays toward a base level
            if food_per_capita > 0.5:
                cp.satisfaction[c] = min(1.0, cp.satisfaction[c] + 0.005 * dt)
            # War weariness
            if cp.war_deaths_this_turn > 0:
                loss_ratio = cp.war_deaths_this_turn / max(cp.total_pop, 1)
                cp.satisfaction[c] = max(0.0, cp.satisfaction[c] - loss_ratio * 5.0)
                cp.radicalization[c] = min(1.0, cp.radicalization[c] + loss_ratio * 2.0)

        # Reset turn counters
        cp.war_deaths_this_turn = 0
        cp.civilian_deaths_this_turn = 0

    world.turn += 1
    return world, feedback


# ═══════════════════════════════════════════════════════════════════════════ #
# Conscription (draws from real population)                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

CONSCRIPTION_LIMITS = {
    "VOLUNTEER_ONLY":       0.02,
    "LIMITED_DRAFT":        0.05,
    "GENERAL_MOBILISATION": 0.12,
    "TOTAL_MOBILISATION":   0.25,
    "SCRAPING_THE_BARREL":  0.40,
}

def conscript(
    cp: ClusterPop,
    count: int,
    law: str = "GENERAL_MOBILISATION",
) -> Tuple[int, str]:
    """
    Conscript `count` people from a cluster. Returns (actually_conscripted, message).
    Draws from unemployed proles first, then employed proles, then outer party.
    Inner Party is EXEMPT.
    """
    max_ratio = CONSCRIPTION_LIMITS.get(law, 0.12)
    max_possible = int(cp.working_age * max_ratio) - cp.total_military
    actual = min(count, max(0, max_possible))
    actual = min(actual, cp.available_for_conscription)

    if actual <= 0:
        return 0, "No available conscripts. Population exhausted."

    remaining = actual

    # Draw from unemployed proles first
    prole_unemp = int(cp.jobs[SocialClass.PROLES.value, JobType.UNEMPLOYED.value])
    from_prole_unemp = min(remaining, prole_unemp)
    cp.jobs[SocialClass.PROLES.value, JobType.UNEMPLOYED.value] -= from_prole_unemp
    remaining -= from_prole_unemp

    # Then unemployed outer party
    if remaining > 0:
        outer_unemp = int(cp.jobs[SocialClass.OUTER_PARTY.value, JobType.UNEMPLOYED.value])
        from_outer_unemp = min(remaining, outer_unemp)
        cp.jobs[SocialClass.OUTER_PARTY.value, JobType.UNEMPLOYED.value] -= from_outer_unemp
        remaining -= from_outer_unemp

    # Then employed prole factory workers (economic penalty!)
    if remaining > 0:
        prole_factory = int(cp.jobs[SocialClass.PROLES.value, JobType.FACTORY_WORKER.value])
        from_factory = min(remaining, prole_factory // 2)  # max half of factory workers
        cp.jobs[SocialClass.PROLES.value, JobType.FACTORY_WORKER.value] -= from_factory
        remaining -= from_factory

    conscripted = actual - remaining

    # Put them into training
    cp.in_training += conscripted
    cp.training_turns_left = 5  # 5 turns basic training

    # Conscription exhaustion
    cp.conscription_exhaustion = min(1.0,
        cp.conscription_exhaustion + conscripted / max(cp.working_age, 1) * 2.0)

    # Morale impact: proles hate conscription
    cp.satisfaction[SocialClass.PROLES.value] = max(0.0,
        cp.satisfaction[SocialClass.PROLES.value] - 0.02 * (conscripted / 1000))
    cp.radicalization[SocialClass.PROLES.value] = min(1.0,
        cp.radicalization[SocialClass.PROLES.value] + 0.01 * (conscripted / 1000))

    return conscripted, f"Conscripted {conscripted:,} from {cp.total_pop:,} population."


# ═══════════════════════════════════════════════════════════════════════════ #
# Observation + Summary                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def pop_summary(world: PopWorld, faction_id: int, cluster_owners: Dict[int, int],
                cluster_names: List[str]) -> str:
    """Generate population summary for turn reports."""
    total_pop = 0
    total_mil = 0
    total_unemployed = 0
    for cp in world.clusters:
        if cluster_owners.get(cp.cluster_id) != faction_id:
            continue
        total_pop += cp.total_pop
        total_mil += cp.total_military
        total_unemployed += cp.total_unemployed

    mil_pct = total_mil / max(total_pop, 1) * 100
    unemp_pct = total_unemployed / max(total_pop, 1) * 100

    parts = [
        f"Population: {total_pop:,}",
        f"Military: {total_mil:,} ({mil_pct:.1f}%)",
        f"Unemployed: {total_unemployed:,} ({unemp_pct:.1f}%)",
    ]

    # Find most populated cities
    cities = []
    for cp in world.clusters:
        if cluster_owners.get(cp.cluster_id) == faction_id:
            name = cluster_names[cp.cluster_id] if cp.cluster_id < len(cluster_names) else f"C{cp.cluster_id}"
            cities.append((cp.total_pop, name))
    cities.sort(reverse=True)
    if cities:
        top = ", ".join(f"{n}({p/1000:.0f}K)" for p, n in cities[:4])
        parts.append(f"Top: {top}")

    return " | ".join(parts)


def pop_obs(world: PopWorld, faction_id: int, cluster_owners: Dict[int, int]) -> np.ndarray:
    """Observation vector for population (for RL or analysis)."""
    obs = []
    for cp in world.clusters:
        if cluster_owners.get(cp.cluster_id) == faction_id:
            obs.extend([
                cp.total_pop / 10_000_000,  # normalize to ~1.0 for London
                cp.military_ratio,
                cp.unemployment_rate,
                cp.prole_fraction,
                float(cp.satisfaction.mean()),
                float(cp.radicalization.mean()),
                cp.conscription_exhaustion,
            ])
    return np.array(obs, dtype=np.float32)
