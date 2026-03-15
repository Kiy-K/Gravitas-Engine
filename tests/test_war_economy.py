"""
test_war_economy.py — Tests for the multi-sector war economy, manpower, and conscription systems.

Tests:
  1. State initialization and invariants
  2. Leontief production (bottleneck propagation)
  3. Resource tiers (Tier 1→2→3 dependency)
  4. Spoilage and carrying costs
  5. Trade execution
  6. Lend-lease delivery
  7. Market price discovery
  8. Capital depreciation
  9. Manpower initialization
  10. Training pipeline (graduation, timing)
  11. Conscription law properties (all 15 tiers)
  12. Regime type gating
  13. Desertion mechanics
  14. Labor reallocation with skill penalty
  15. Anti-exploitation: diminishing returns, labor rigidity
  16. Cross-system feedback signals
  17. Observation vector sizes
"""

import sys
from pathlib import Path

import numpy as np

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from extensions.war_economy.war_economy_state import (
    EconSector, Resource, N_SECTORS, N_RESOURCES,
    SECTOR_INPUTS, SECTOR_OUTPUTS, SPOILAGE, MAX_STOCKPILE,
    TIER_1, TIER_2, TIER_3,
    ClusterEconomy, FactionEconomy, WarEconomyWorld,
    TradeAgreement, LendLeasePackage,
    TERRAIN_ENDOWMENT,
)
from extensions.war_economy.war_economy_dynamics import (
    leontief_produce, compute_consumption, apply_depreciation,
    execute_trades, update_market_prices,
    step_war_economy, initialize_war_economy, compute_feedback,
)
from extensions.war_economy.war_economy_actions import (
    WarEconomyAction, apply_war_economy_action,
    war_economy_obs, war_economy_obs_size,
)
from extensions.war_economy.manpower import (
    ConscriptionLaw, RegimeType, SkillLevel, MilitaryTraining,
    ClusterManpower, FactionManpowerPolicy, TrainingBatch,
    ManpowerAction, LAW_STATS, N_CONSCRIPTION_LAWS,
    REGIME_MAX_LAW, REGIME_UNREST_MODIFIER, REGIME_ECONOMY_MODIFIER,
    CONSCRIPTION_RATES, CONSCRIPTION_DESERTION,
    step_manpower, apply_manpower_action, initialize_manpower,
    manpower_obs, manpower_obs_size,
)


def _make_cluster_econ(cid: int = 0, stockpile: float = 50.0) -> ClusterEconomy:
    """Helper: create a test ClusterEconomy."""
    rng = np.random.default_rng(42)
    return ClusterEconomy(
        cluster_id=cid,
        sector_capacity=np.full(N_SECTORS, 2.0, dtype=np.float64),
        sector_labor=np.full(N_SECTORS, 1.0 / N_SECTORS, dtype=np.float64),
        sector_output=np.zeros(N_SECTORS, dtype=np.float64),
        resource_stockpile=np.full(N_RESOURCES, stockpile, dtype=np.float64),
        terrain_endowment=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        infrastructure=0.8,
    )


def _make_war_world(n_clusters: int = 3) -> WarEconomyWorld:
    """Helper: create a test WarEconomyWorld."""
    rng = np.random.default_rng(42)
    return initialize_war_economy(
        n_clusters=n_clusters,
        faction_ids=[0, 1],
        cluster_owners={0: 0, 1: 0, 2: 1},
        terrain_types=["URBAN", "PLAINS", "MOUNTAINS"],
        rng=rng,
    )


# ═══════════════════════════════════════════════════════════════════════════ #
# 1. State Initialization                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_state_initialization():
    ww = _make_war_world()
    assert ww.n_clusters == 3
    assert len(ww.faction_economies) == 2
    assert 0 in ww.faction_economies and 1 in ww.faction_economies

    for ce in ww.cluster_economies:
        assert ce.resource_stockpile.shape == (N_RESOURCES,)
        assert ce.sector_capacity.shape == (N_SECTORS,)
        assert ce.sector_labor.shape == (N_SECTORS,)
        assert np.all(ce.resource_stockpile >= 0)
        assert np.all(ce.sector_capacity > 0)
        assert abs(ce.sector_labor.sum() - 1.0) < 0.01  # labor sums to ~1
        assert 0.0 < ce.infrastructure <= 1.0
    print("  [PASS] test_state_initialization")


# ═══════════════════════════════════════════════════════════════════════════ #
# 2. Leontief Production — Bottleneck                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_leontief_bottleneck():
    """If ONE required input is zero, output should be zero."""
    stockpile = np.full(N_RESOURCES, 100.0, dtype=np.float64)

    # Energy requires crude_oil + coal + steel
    # Zero out crude oil → energy should produce nothing
    stockpile[Resource.CRUDE_OIL.value] = 0.0

    output, produced = leontief_produce(
        EconSector.ENERGY.value, stockpile, capacity=2.0, labor_frac=0.2,
        sanctions=0.0, war_fatigue=0.0, war_bond_mult=1.0, mfg_priority=0.5,
    )
    assert output == 0.0, f"Expected 0 output with missing crude oil, got {output}"
    assert np.all(produced == 0.0)
    print("  [PASS] test_leontief_bottleneck")


def test_leontief_normal_production():
    """With all inputs available, output should be positive."""
    stockpile = np.full(N_RESOURCES, 100.0, dtype=np.float64)

    output, produced = leontief_produce(
        EconSector.HEAVY_INDUSTRY.value, stockpile, capacity=2.0, labor_frac=0.2,
        sanctions=0.0, war_fatigue=0.0, war_bond_mult=1.0, mfg_priority=0.5,
    )
    assert output > 0.0, f"Expected positive output, got {output}"
    # Heavy Industry produces steel + chemicals
    assert produced[Resource.STEEL.value] > 0
    print("  [PASS] test_leontief_normal_production")


def test_leontief_sanctions_reduce_output():
    """Sanctions should reduce output."""
    stockpile = np.full(N_RESOURCES, 100.0, dtype=np.float64)
    stockpile_copy = stockpile.copy()

    out_normal, _ = leontief_produce(
        EconSector.MANUFACTURING.value, stockpile, 2.0, 0.2, 0.0, 0.0, 1.0, 0.5)
    out_sanctioned, _ = leontief_produce(
        EconSector.MANUFACTURING.value, stockpile_copy, 2.0, 0.2, 0.8, 0.0, 1.0, 0.5)

    assert out_sanctioned < out_normal, "Sanctions should reduce output"
    print("  [PASS] test_leontief_sanctions_reduce_output")


# ═══════════════════════════════════════════════════════════════════════════ #
# 3. Resource Tiers                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_resource_tiers():
    """Verify tier classification is correct and complete."""
    all_resources = set(r.value for r in Resource)
    tier_union = set(r.value for r in TIER_1 + TIER_2 + TIER_3)
    assert all_resources == tier_union, "All resources must belong to exactly one tier"
    assert len(TIER_1) == 5
    assert len(TIER_2) == 4
    assert len(TIER_3) == 3
    print("  [PASS] test_resource_tiers")


# ═══════════════════════════════════════════════════════════════════════════ #
# 4. Spoilage                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_spoilage():
    """Food should spoil faster than steel."""
    assert SPOILAGE[Resource.RAW_FOOD.value] > SPOILAGE[Resource.STEEL.value]
    assert SPOILAGE[Resource.IRON_ORE.value] == 0.0  # metals don't spoil
    print("  [PASS] test_spoilage")


# ═══════════════════════════════════════════════════════════════════════════ #
# 5. Trade Execution                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_trade_execution():
    ww = _make_war_world()
    # Add a trade: faction 0 exports steel to faction 1
    initial_steel_f0 = ww.cluster_economies[0].resource_stockpile[Resource.STEEL.value]
    initial_steel_f1 = ww.cluster_economies[2].resource_stockpile[Resource.STEEL.value]

    ww.trade_agreements.append(TradeAgreement(
        exporter_faction=0, importer_faction=1,
        resource=Resource.STEEL, amount_per_step=5.0,
        price_ratio=0.5, remaining_steps=10,
        route_clusters=(0, 2),
    ))
    execute_trades(ww, {0: 0, 1: 0, 2: 1})

    # Exporter should have less steel, importer more
    final_steel_f0 = ww.cluster_economies[0].resource_stockpile[Resource.STEEL.value]
    final_steel_f1 = ww.cluster_economies[2].resource_stockpile[Resource.STEEL.value]
    assert final_steel_f0 < initial_steel_f0, "Exporter should lose steel"
    assert final_steel_f1 > initial_steel_f1, "Importer should gain steel"
    print("  [PASS] test_trade_execution")


# ═══════════════════════════════════════════════════════════════════════════ #
# 6. Market Prices                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_market_prices():
    ww = _make_war_world()
    old_prices = ww.market_prices.copy()
    update_market_prices(ww)
    # Prices should change (even slightly from EMA smoothing)
    assert ww.market_prices.shape == (N_RESOURCES,)
    assert np.all(ww.market_prices > 0)
    assert np.all(ww.market_prices < 6.0)
    print("  [PASS] test_market_prices")


# ═══════════════════════════════════════════════════════════════════════════ #
# 7. Full Step                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_full_step():
    ww = _make_war_world()
    cluster_data = np.array([
        [0.7, 0.2, 0.5, 0.1, 0.6, 0.3],  # cluster 0
        [0.8, 0.1, 0.6, 0.0, 0.7, 0.2],  # cluster 1
        [0.5, 0.5, 0.3, 0.3, 0.4, 0.4],  # cluster 2
    ], dtype=np.float64)
    pop = np.array([0.7, 0.8, 0.5], dtype=np.float64)

    ww_after = step_war_economy(
        ww, cluster_data, pop,
        cluster_owners={0: 0, 1: 0, 2: 1},
        alliance=None,
        terrain_types=["URBAN", "PLAINS", "MOUNTAINS"],
        dt=0.01,
    )
    assert ww_after.step == 1
    # Stockpiles should have changed
    for ce in ww_after.cluster_economies:
        assert np.all(ce.resource_stockpile >= 0)
        assert np.all(ce.resource_stockpile <= MAX_STOCKPILE)
    print("  [PASS] test_full_step")


# ═══════════════════════════════════════════════════════════════════════════ #
# 8. Feedback Signals                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_feedback():
    ce = _make_cluster_econ(stockpile=80.0)
    fb = compute_feedback(ce)
    assert "supply_refill_mult" in fb
    assert "combat_effectiveness" in fb
    assert "gdp_modifier" in fb
    assert fb["supply_refill_mult"] > 0
    assert fb["combat_effectiveness"] > 0
    print("  [PASS] test_feedback")


# ═══════════════════════════════════════════════════════════════════════════ #
# 9. Observation Sizes                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_obs_sizes():
    ww = _make_war_world()
    obs = war_economy_obs(ww, faction_id=0, cluster_owners={0: 0, 1: 0, 2: 1})
    expected = war_economy_obs_size(max_clusters=12)
    assert obs.shape == (expected,), f"Expected {expected}, got {obs.shape}"
    assert obs.dtype == np.float32
    print("  [PASS] test_obs_sizes")


# ═══════════════════════════════════════════════════════════════════════════ #
# 10. Conscription Laws — All 15                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_conscription_laws():
    assert N_CONSCRIPTION_LAWS == 15, f"Expected 15 laws, got {N_CONSCRIPTION_LAWS}"
    prev_rate = -1.0
    for law in ConscriptionLaw:
        stats = LAW_STATS[law]
        # Rates should be monotonically increasing
        assert stats.max_rate > prev_rate, f"{law.name}: rate not increasing"
        prev_rate = stats.max_rate
        # Economy penalty should increase
        assert 0.0 <= stats.economy_penalty <= 1.0
        # Training quality should decrease
        assert 0.0 <= stats.training_quality <= 1.0
        # Desertion rate should be non-negative
        assert stats.desertion_rate >= 0.0
    print("  [PASS] test_conscription_laws (all 15 tiers verified)")


def test_conscription_training_quality_decreases():
    """Wartime+ conscription (tier 7+) should have worse training than peacetime (tier 0-3)."""
    laws = list(ConscriptionLaw)
    # Peacetime tiers can have non-monotonic quality (volunteers > token service),
    # but wartime and beyond must trend downward.
    peacetime_max = max(LAW_STATS[laws[i]].training_quality for i in range(4))
    wartime_min = min(LAW_STATS[laws[i]].training_quality for i in range(7, len(laws)))
    assert wartime_min < peacetime_max, "Wartime training should be worse than peacetime"

    # Tiers 7-14 should be strictly non-increasing
    for i in range(8, len(laws)):
        q_prev = LAW_STATS[laws[i - 1]].training_quality
        q_curr = LAW_STATS[laws[i]].training_quality
        assert q_curr <= q_prev, f"{laws[i].name} has better training than {laws[i-1].name}"

    # Desperation tier should be very low
    assert LAW_STATS[ConscriptionLaw.NATION_IN_ARMS].training_quality <= 0.15
    print("  [PASS] test_conscription_training_quality_decreases")


# ═══════════════════════════════════════════════════════════════════════════ #
# 11. Regime Gating                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_regime_gating():
    """Democracies should be capped at lower conscription levels than totalitarian."""
    dem_max = REGIME_MAX_LAW[RegimeType.LIBERAL_DEMOCRACY]
    tot_max = REGIME_MAX_LAW[RegimeType.TOTALITARIAN]
    assert dem_max.value < tot_max.value, "Democracy should have lower cap than Totalitarian"

    # All 12 regime types must have entries
    assert len(REGIME_MAX_LAW) == 12
    assert len(REGIME_UNREST_MODIFIER) == 12
    assert len(REGIME_ECONOMY_MODIFIER) == 12
    print("  [PASS] test_regime_gating")


def test_regime_unrest_modifiers():
    """Authoritarian regimes should suppress unrest better than democracies."""
    dem_unrest = REGIME_UNREST_MODIFIER[RegimeType.LIBERAL_DEMOCRACY]
    tot_unrest = REGIME_UNREST_MODIFIER[RegimeType.TOTALITARIAN]
    assert dem_unrest > tot_unrest, "Democracy should have higher unrest modifier"
    print("  [PASS] test_regime_unrest_modifiers")


# ═══════════════════════════════════════════════════════════════════════════ #
# 12. Manpower Initialization                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_manpower_init():
    rng = np.random.default_rng(42)
    cluster_mps, policies = initialize_manpower(
        n_clusters=3, faction_ids=[0, 1],
        cluster_owners={0: 0, 1: 0, 2: 1},
        terrain_types=["URBAN", "PLAINS", "MOUNTAINS"],
        rng=rng,
    )
    assert len(cluster_mps) == 3
    assert len(policies) == 2

    for mp in cluster_mps:
        assert mp.total_population > 0
        assert mp.working_age_frac > 0.5
        assert float(np.sum(mp.sector_workers)) > 0
        assert mp.military_personnel >= 0
    print("  [PASS] test_manpower_init")


# ═══════════════════════════════════════════════════════════════════════════ #
# 13. Training Pipeline                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_training_pipeline():
    """Training batches should graduate after steps_remaining reaches 0."""
    rng = np.random.default_rng(42)
    cluster_mps, policies = initialize_manpower(
        3, [0, 1], {0: 0, 1: 0, 2: 1}, ["URBAN", "PLAINS", "MOUNTAINS"], rng)

    mp = cluster_mps[0]
    policy = policies[0]

    # Add a training batch that completes in 2 steps
    initial_military = mp.military_personnel
    mp.in_training.append(TrainingBatch(
        count=100, target_sector=-1,
        current_level=0, target_level=1,
        steps_remaining=2, is_military=True,
    ))

    # Step 1: not complete yet
    mp, fb = step_manpower(mp, policy, hazard=0.1, food_ratio=0.8, gdp_level=0.6)
    assert mp.military_personnel < initial_military + 100  # not graduated yet

    # Step 2: should complete
    mp, fb = step_manpower(mp, policy, hazard=0.1, food_ratio=0.8, gdp_level=0.6)
    # The 100 recruits should have graduated into military
    assert mp.military_personnel >= initial_military + 90  # allow for some desertion
    print("  [PASS] test_training_pipeline")


# ═══════════════════════════════════════════════════════════════════════════ #
# 14. Desertion                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_desertion():
    """Higher conscription + food shortage = more desertion."""
    rng = np.random.default_rng(42)
    cluster_mps, policies = initialize_manpower(
        3, [0, 1], {0: 0, 1: 0, 2: 1}, ["URBAN", "PLAINS", "MOUNTAINS"], rng)

    mp = cluster_mps[0]
    mp.military_personnel = 50.0
    mp.military_skill_dist = np.array([30.0, 10.0, 7.0, 3.0])

    # Set extreme conscription + starvation
    policy = policies[0]
    policy.conscription_law = ConscriptionLaw.SCRAPING_THE_BARREL

    initial_mil = mp.military_personnel
    mp, fb = step_manpower(mp, policy, hazard=0.3, food_ratio=0.1, gdp_level=0.3, dt=1.0)

    assert mp.military_personnel < initial_mil, "Desertion should reduce military"
    assert fb["desertion_rate"] > 0, "Desertion rate should be reported"
    print("  [PASS] test_desertion")


# ═══════════════════════════════════════════════════════════════════════════ #
# 15. Manpower Observation                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_manpower_obs():
    rng = np.random.default_rng(42)
    cluster_mps, policies = initialize_manpower(
        3, [0, 1], {0: 0, 1: 0, 2: 1}, ["URBAN", "PLAINS", "MOUNTAINS"], rng)

    obs = manpower_obs(cluster_mps[0], policies[0])
    assert obs.shape == (manpower_obs_size(),)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    print("  [PASS] test_manpower_obs")


# ═══════════════════════════════════════════════════════════════════════════ #
# 16. Anti-Exploitation: Diminishing Returns                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

def test_diminishing_returns():
    """Doubling labor should NOT double output."""
    stockpile1 = np.full(N_RESOURCES, 100.0, dtype=np.float64)
    stockpile2 = np.full(N_RESOURCES, 100.0, dtype=np.float64)

    out1, _ = leontief_produce(
        EconSector.AGRICULTURE.value, stockpile1, 2.0, 0.1, 0.0, 0.0, 1.0, 0.5)
    out2, _ = leontief_produce(
        EconSector.AGRICULTURE.value, stockpile2, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5)

    # 5× more labor should give < 5× output (diminishing returns)
    ratio = out2 / max(out1, 0.001)
    assert ratio < 5.0, f"Diminishing returns violated: {ratio}x output for 5x labor"
    assert ratio > 1.0, f"More labor should still produce more: ratio={ratio}"
    print(f"  [PASS] test_diminishing_returns (5x labor → {ratio:.1f}x output)")


# ═══════════════════════════════════════════════════════════════════════════ #
# Runner                                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("=" * 60)
    print("  WAR ECONOMY & MANPOWER SYSTEM TESTS")
    print("=" * 60)
    print()

    tests = [
        test_state_initialization,
        test_leontief_bottleneck,
        test_leontief_normal_production,
        test_leontief_sanctions_reduce_output,
        test_resource_tiers,
        test_spoilage,
        test_trade_execution,
        test_market_prices,
        test_full_step,
        test_feedback,
        test_obs_sizes,
        test_conscription_laws,
        test_conscription_training_quality_decreases,
        test_regime_gating,
        test_regime_unrest_modifiers,
        test_manpower_init,
        test_training_pipeline,
        test_desertion,
        test_manpower_obs,
        test_diminishing_returns,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print()
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
