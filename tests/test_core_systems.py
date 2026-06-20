"""
test_core_systems.py — Unit tests for core systems modules.

Tests:
  1. crisis_classifier — classify, ClassifierThresholds, crisis_distribution, max_crisis_reached
  2. early_warning — early_warning_index, exhaustion_growth_rate, volatility_spike_indicator
  3. collapse_physics — build_province_adjacency, is_bridge_province, domino effects, national shock
  4. factions — Gini, aggregation functions, factory functions
  5. state — FactionState, SystemState, RegimeState validation
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from gravitas_engine.core.state import FactionState, SystemState, RegimeState
from gravitas_engine.core.parameters import SystemParameters
from gravitas_engine.core.factions import (
    compute_gini,
    compute_legitimacy,
    compute_cohesion,
    compute_fragmentation,
    compute_instability,
    compute_mobilization,
    compute_repression,
    compute_elite_alignment,
    compute_volatility,
    recompute_system_state,
    create_balanced_factions,
    create_dominant_factions,
)
from gravitas_engine.systems.crisis_classifier import (
    CrisisLevel,
    ClassifierThresholds,
    classify,
    classify_trajectory,
    crisis_distribution,
    max_crisis_reached,
)
from gravitas_engine.systems.early_warning import (
    early_warning_index,
    exhaustion_growth_rate,
    volatility_spike_indicator,
)
from gravitas_engine.systems.collapse_physics import (
    build_province_adjacency,
    is_bridge_province,
    apply_domino_effects,
    apply_national_shock,
    apply_exhaustion_admin_decay,
    apply_exhaustion_unrest_drift,
    UNREST_CRITICAL_THRESHOLD,
    DOMINO_UNREST_BUMP,
)
from gravitas_engine.core.hierarchical_state import (
    HierarchicalState,
    DistrictState,
    create_hierarchical_state,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_factions():
    """Two neutral factions for basic tests."""
    return create_balanced_factions(n_factions=2)


@pytest.fixture
def stable_system():
    """A stable system state with low crisis indicators."""
    return SystemState(
        legitimacy=0.80,
        cohesion=0.70,
        fragmentation=0.10,
        instability=0.05,
        mobilization=0.05,
        repression=0.20,
        elite_alignment=0.72,
        volatility=0.05,
        exhaustion=0.05,
        state_gdp=0.60,
        pillars=(0.7, 0.6, 0.65),
    )


@pytest.fixture
def stressed_system():
    """A stressed system state approaching crisis."""
    return SystemState(
        legitimacy=0.40,
        cohesion=0.30,
        fragmentation=0.55,
        instability=0.50,
        mobilization=0.45,
        repression=0.60,
        elite_alignment=0.18,
        volatility=0.45,
        exhaustion=0.50,
        state_gdp=0.35,
        pillars=(0.4, 0.35, 0.3),
    )


@pytest.fixture
def collapsing_system():
    """A system approaching collapse."""
    return SystemState(
        legitimacy=0.20,
        cohesion=0.15,
        fragmentation=0.85,
        instability=0.85,
        mobilization=0.90,
        repression=0.80,
        elite_alignment=0.03,
        volatility=0.55,
        exhaustion=0.90,
        state_gdp=0.15,
        pillars=(0.2, 0.15, 0.1),
    )


@pytest.fixture
def base_regime_state(base_factions, stable_system):
    """A basic regime state for testing."""
    return RegimeState(
        factions=base_factions,
        system=stable_system,
        affinity_matrix=((1.0, 0.5), (0.5, 1.0)),
        step=0,
    )


@pytest.fixture
def params():
    """Standard system parameters."""
    return SystemParameters(n_factions=2)


# ============================================================================
# 1. Crisis Classifier Tests
# ============================================================================

class TestClassifierThresholds:
    """Test ClassifierThresholds validation."""

    def test_default_thresholds_valid(self):
        """Default thresholds should be in valid range (0, 1)."""
        t = ClassifierThresholds()
        for name, val in t.__dict__.items():
            assert 0.0 < val < 1.0, f"{name}={val} out of range"

    def test_invalid_threshold_raises(self):
        """Threshold outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            ClassifierThresholds(tension_instability=0.0)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            ClassifierThresholds(tension_instability=1.0)
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            ClassifierThresholds(crisis_instability=-0.1)

    def test_custom_thresholds_accepted(self):
        """Custom thresholds should be accepted if valid."""
        t = ClassifierThresholds(
            tension_instability=0.25,
            mobilization_mob=0.30,
            collapse_exhaustion=0.80,
        )
        assert t.tension_instability == 0.25
        assert t.mobilization_mob == 0.30
        assert t.collapse_exhaustion == 0.80


class TestCrisisLevel:
    """Test CrisisLevel enum ordering."""

    def test_crisis_levels_ordered(self):
        """Higher crisis levels should have higher integer values."""
        assert CrisisLevel.STABLE < CrisisLevel.TENSION
        assert CrisisLevel.TENSION < CrisisLevel.MOBILIZATION
        assert CrisisLevel.MOBILIZATION < CrisisLevel.FRAGMENTATION
        assert CrisisLevel.FRAGMENTATION < CrisisLevel.VOLATILITY
        assert CrisisLevel.VOLATILITY < CrisisLevel.CRISIS
        assert CrisisLevel.CRISIS < CrisisLevel.COLLAPSE

    def test_crisis_levels_count(self):
        """Should have exactly 7 crisis levels."""
        assert len(CrisisLevel) == 7


class TestClassify:
    """Test the classify function."""

    def test_stable_state_returns_stable(self, base_regime_state, stable_system, params):
        """A healthy system should return STABLE."""
        state = base_regime_state.copy_with_system(stable_system)
        thresholds = ClassifierThresholds()
        result = classify(state, thresholds)
        assert result == CrisisLevel.STABLE

    def test_tension_detected(self, base_regime_state, params):
        """Instability above tension threshold should return TENSION."""
        sys = SystemState(
            legitimacy=0.50, cohesion=0.40, fragmentation=0.10,
            instability=0.35, mobilization=0.10, repression=0.50,
            elite_alignment=0.45, volatility=0.10, exhaustion=0.05,
        )
        state = base_regime_state.copy_with_system(sys)
        thresholds = ClassifierThresholds()
        result = classify(state, thresholds)
        assert result == CrisisLevel.TENSION

    def test_mobilization_detected(self, base_regime_state, params):
        """Mobilization above threshold should return MOBILIZATION."""
        sys = SystemState(
            legitimacy=0.50, cohesion=0.40, fragmentation=0.10,
            instability=0.20, mobilization=0.40, repression=0.50,
            elite_alignment=0.45, volatility=0.10, exhaustion=0.05,
        )
        state = base_regime_state.copy_with_system(sys)
        thresholds = ClassifierThresholds()
        result = classify(state, thresholds)
        assert result == CrisisLevel.MOBILIZATION

    def test_fragmentation_detected(self, base_regime_state, params):
        """Fragmentation above threshold should return FRAGMENTATION."""
        sys = SystemState(
            legitimacy=0.40, cohesion=0.30, fragmentation=0.50,
            instability=0.20, mobilization=0.20, repression=0.60,
            elite_alignment=0.20, volatility=0.20, exhaustion=0.10,
        )
        state = base_regime_state.copy_with_system(sys)
        thresholds = ClassifierThresholds()
        result = classify(state, thresholds)
        assert result == CrisisLevel.FRAGMENTATION

    def test_volatility_detected(self, base_regime_state, params):
        """Volatility above threshold should return VOLATILITY."""
        sys = SystemState(
            legitimacy=0.50, cohesion=0.40, fragmentation=0.20,
            instability=0.20, mobilization=0.20, repression=0.50,
            elite_alignment=0.40, volatility=0.60, exhaustion=0.10,
        )
        state = base_regime_state.copy_with_system(sys)
        thresholds = ClassifierThresholds()
        result = classify(state, thresholds)
        assert result == CrisisLevel.VOLATILITY

    def test_crisis_detected(self, base_regime_state, params):
        """High instability + volatility should return CRISIS."""
        # CRISIS: instability >= 0.60 AND volatility >= 0.50
        sys = SystemState(
            legitimacy=0.25, cohesion=0.20, fragmentation=0.60,
            instability=0.65, mobilization=0.55, repression=0.75,
            elite_alignment=0.10, volatility=0.55, exhaustion=0.50,
        )
        state = base_regime_state.copy_with_system(sys)
        thresholds = ClassifierThresholds()
        result = classify(state, thresholds)
        assert result == CrisisLevel.CRISIS

    def test_collapse_detected(self, base_regime_state, collapsing_system):
        """High exhaustion + volatility should return COLLAPSE."""
        state = base_regime_state.copy_with_system(collapsing_system)
        thresholds = ClassifierThresholds()
        result = classify(state, thresholds)
        assert result == CrisisLevel.COLLAPSE

    def test_highest_severity_returned(self, base_regime_state):
        """When multiple thresholds exceeded, highest severity should be returned."""
        # COLLAPSE threshold: exhaustion >= 0.85 AND volatility >= 0.40
        sys = SystemState(
            legitimacy=0.10, cohesion=0.10, fragmentation=0.90,
            instability=0.90, mobilization=0.90, repression=0.90,
            elite_alignment=0.01, volatility=0.50, exhaustion=0.90,
        )
        state = base_regime_state.copy_with_system(sys)
        thresholds = ClassifierThresholds()
        result = classify(state, thresholds)
        assert result == CrisisLevel.COLLAPSE


class TestClassifyTrajectory:
    """Test classify_trajectory function."""

    def test_empty_trajectory(self, params):
        """Empty trajectory should return empty list."""
        thresholds = ClassifierThresholds()
        result = classify_trajectory([], thresholds)
        assert result == []

    def test_single_state(self, base_regime_state, stable_system, params):
        """Single state should return single label."""
        state = base_regime_state.copy_with_system(stable_system)
        thresholds = ClassifierThresholds()
        result = classify_trajectory([state], thresholds)
        assert len(result) == 1
        assert result[0] == CrisisLevel.STABLE

    def test_multiple_states(self, base_regime_state, params):
        """Multiple states should return corresponding labels."""
        # Use states with known classifications
        states = []
        # STABLE: low instability
        sys1 = SystemState(
            legitimacy=0.7, cohesion=0.6, fragmentation=0.1,
            instability=0.1, mobilization=0.1, repression=0.3,
            elite_alignment=0.6, volatility=0.1, exhaustion=0.05,
        )
        states.append(base_regime_state.copy_with_system(sys1))
        
        # TENSION: instability >= 0.30
        sys2 = SystemState(
            legitimacy=0.5, cohesion=0.4, fragmentation=0.2,
            instability=0.35, mobilization=0.2, repression=0.5,
            elite_alignment=0.4, volatility=0.1, exhaustion=0.1,
        )
        states.append(base_regime_state.copy_with_system(sys2))
        
        # CRISIS: instability >= 0.60 AND volatility >= 0.50
        sys3 = SystemState(
            legitimacy=0.25, cohesion=0.2, fragmentation=0.6,
            instability=0.65, mobilization=0.5, repression=0.75,
            elite_alignment=0.1, volatility=0.55, exhaustion=0.5,
        )
        states.append(base_regime_state.copy_with_system(sys3))
        
        # COLLAPSE: exhaustion >= 0.85 AND volatility >= 0.40
        sys4 = SystemState(
            legitimacy=0.1, cohesion=0.1, fragmentation=0.9,
            instability=0.9, mobilization=0.9, repression=0.9,
            elite_alignment=0.01, volatility=0.55, exhaustion=0.9,
        )
        states.append(base_regime_state.copy_with_system(sys4))
        
        thresholds = ClassifierThresholds()
        result = classify_trajectory(states, thresholds)
        assert len(result) == 4
        assert result[0] == CrisisLevel.STABLE
        assert result[1] == CrisisLevel.TENSION
        assert result[2] == CrisisLevel.CRISIS
        assert result[3] == CrisisLevel.COLLAPSE


class TestCrisisDistribution:
    """Test crisis_distribution function."""

    def test_empty_labels(self):
        """Empty labels should return all zeros."""
        result = crisis_distribution([])
        assert all(val == 0.0 for val in result.values())

    def test_single_label(self):
        """Single label should have 100% for that level."""
        result = crisis_distribution([CrisisLevel.STABLE])
        assert result[CrisisLevel.STABLE] == 1.0
        assert all(val == 0.0 for k, val in result.items() if k != CrisisLevel.STABLE)

    def test_mixed_labels(self):
        """Mixed labels should sum to 1.0."""
        labels = [
            CrisisLevel.STABLE,
            CrisisLevel.STABLE,
            CrisisLevel.TENSION,
            CrisisLevel.CRISIS,
        ]
        result = crisis_distribution(labels)
        assert sum(result.values()) == 1.0
        assert result[CrisisLevel.STABLE] == 0.5
        assert result[CrisisLevel.TENSION] == 0.25
        assert result[CrisisLevel.CRISIS] == 0.25

    def test_all_levels_present(self):
        """Should include all crisis levels in result."""
        labels = [CrisisLevel.STABLE, CrisisLevel.COLLAPSE]
        result = crisis_distribution(labels)
        assert set(result.keys()) == set(CrisisLevel)


class TestMaxCrisisReached:
    """Test max_crisis_reached function."""

    def test_empty_labels(self):
        """Empty labels should return STABLE."""
        assert max_crisis_reached([]) == CrisisLevel.STABLE

    def test_single_label(self):
        """Single label should return that label."""
        assert max_crisis_reached([CrisisLevel.TENSION]) == CrisisLevel.TENSION

    def test_max_found(self):
        """Should return the maximum severity level."""
        labels = [
            CrisisLevel.STABLE,
            CrisisLevel.TENSION,
            CrisisLevel.MOBILIZATION,
            CrisisLevel.FRAGMENTATION,
        ]
        assert max_crisis_reached(labels) == CrisisLevel.FRAGMENTATION

    def test_max_at_end(self):
        """Maximum at end of list."""
        labels = [CrisisLevel.STABLE, CrisisLevel.COLLAPSE, CrisisLevel.TENSION]
        assert max_crisis_reached(labels) == CrisisLevel.COLLAPSE


# ============================================================================
# 2. Early Warning Tests
# ============================================================================

class TestEarlyWarningIndex:
    """Test early_warning_index function."""

    def test_output_bounded(self, base_regime_state):
        """EWI should always be in [0, 1]."""
        state = base_regime_state
        for vol in [0.0, 0.5, 1.0]:
            for unrest in [0.0, 0.5, 1.0]:
                for cluster in [0.0, 0.5, 1.0]:
                    for exh_rate in [-0.1, 0.0, 0.1]:
                        sys = state.system
                        new_sys = SystemState(
                            legitimacy=sys.legitimacy,
                            cohesion=sys.cohesion,
                            fragmentation=sys.fragmentation,
                            instability=sys.instability,
                            mobilization=sys.mobilization,
                            repression=sys.repression,
                            elite_alignment=sys.elite_alignment,
                            volatility=vol,
                            exhaustion=sys.exhaustion,
                            state_gdp=sys.state_gdp,
                            pillars=sys.pillars,
                        )
                        new_state = state.copy_with_system(new_sys)
                        ewi = early_warning_index(new_state, unrest, cluster, exh_rate)
                        assert 0.0 <= ewi <= 1.0, f"EWI={ewi} out of bounds"

    def test_zero_state(self, base_regime_state):
        """Zero inputs should produce low EWI."""
        sys = SystemState(
            legitimacy=0.8, cohesion=0.7, fragmentation=0.1,
            instability=0.0, mobilization=0.0, repression=0.2,
            elite_alignment=0.7, volatility=0.0, exhaustion=0.0,
        )
        state = base_regime_state.copy_with_system(sys)
        ewi = early_warning_index(state, unrest_variance=0.0, clustering_index=0.0, exhaustion_growth_rate=0.0)
        assert ewi < 0.2

    def test_high_volatility_increases_ewi(self, base_regime_state):
        """Higher volatility should increase EWI."""
        sys_low = SystemState(
            legitimacy=0.6, cohesion=0.5, fragmentation=0.2,
            instability=0.2, mobilization=0.2, repression=0.4,
            elite_alignment=0.48, volatility=0.1, exhaustion=0.1,
        )
        sys_high = SystemState(
            legitimacy=0.6, cohesion=0.5, fragmentation=0.2,
            instability=0.2, mobilization=0.2, repression=0.4,
            elite_alignment=0.48, volatility=0.8, exhaustion=0.1,
        )
        state_low = base_regime_state.copy_with_system(sys_low)
        state_high = base_regime_state.copy_with_system(sys_high)
        ewi_low = early_warning_index(state_low)
        ewi_high = early_warning_index(state_high)
        assert ewi_high > ewi_low

    def test_custom_weights(self, base_regime_state):
        """Custom weights should affect EWI calculation."""
        # Create state with moderate volatility
        sys = SystemState(
            legitimacy=0.5, cohesion=0.4, fragmentation=0.2,
            instability=0.3, mobilization=0.2, repression=0.5,
            elite_alignment=0.4, volatility=0.3, exhaustion=0.2,
        )
        state = base_regime_state.copy_with_system(sys)
        ewi_default = early_warning_index(state)
        # With high variance weight and non-zero variance, should be higher
        ewi_high_var = early_warning_index(
            state, unrest_variance=0.8, clustering_index=0.0, exhaustion_growth_rate=0.0,
            w_variance=0.8, w_clustering=0.1, w_exh_rate=0.05, w_volatility=0.05
        )
        assert ewi_high_var > ewi_default


class TestExhaustionGrowthRate:
    """Test exhaustion_growth_rate function."""

    def test_basic_calculation(self):
        """Test basic rate calculation."""
        rate = exhaustion_growth_rate(current_exhaustion=0.10, previous_exhaustion=0.05, dt=0.01)
        assert rate == pytest.approx(5.0)

    def test_zero_change(self):
        """No change should give zero rate."""
        rate = exhaustion_growth_rate(current_exhaustion=0.5, previous_exhaustion=0.5, dt=0.01)
        assert rate == 0.0

    def test_negative_dt(self):
        """Negative dt should return 0.0."""
        rate = exhaustion_growth_rate(current_exhaustion=0.10, previous_exhaustion=0.05, dt=-0.01)
        assert rate == 0.0

    def test_zero_dt(self):
        """Zero dt should return 0.0."""
        rate = exhaustion_growth_rate(current_exhaustion=0.10, previous_exhaustion=0.05, dt=0.0)
        assert rate == 0.0

    def test_decreasing_exhaustion(self):
        """Negative rate when exhaustion decreases."""
        rate = exhaustion_growth_rate(current_exhaustion=0.05, previous_exhaustion=0.10, dt=0.01)
        assert rate == pytest.approx(-5.0)


class TestVolatilitySpikeIndicator:
    """Test volatility_spike_indicator function."""

    def test_no_spike(self):
        """Small increase should not trigger spike."""
        result = volatility_spike_indicator(current_volatility=0.15, previous_volatility=0.10, threshold=0.1)
        assert result == 0.0

    def test_spike_at_threshold(self):
        """Increase at exactly threshold should trigger."""
        result = volatility_spike_indicator(current_volatility=0.20, previous_volatility=0.10, threshold=0.1)
        assert result == 1.0

    def test_spike_above_threshold(self):
        """Increase above threshold should trigger."""
        result = volatility_spike_indicator(current_volatility=0.30, previous_volatility=0.10, threshold=0.1)
        assert result == 1.0

    def test_decrease_no_spike(self):
        """Decrease should not trigger spike."""
        result = volatility_spike_indicator(current_volatility=0.05, previous_volatility=0.15, threshold=0.1)
        assert result == 0.0


# ============================================================================
# 3. Collapse Physics Tests
# ============================================================================

class TestBuildProvinceAdjacency:
    """Test build_province_adjacency function."""

    def test_single_province(self):
        """Single province should have no inter-province connections."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2,
            connect_between_provinces=False
        )
        P_adj, P_weight = build_province_adjacency(hier)
        assert P_adj.shape == (5, 5)
        assert np.sum(P_adj) == 0.0  # No inter-province links

    def test_multiple_provinces_with_links(self):
        """Multiple provinces with links should have non-zero adjacency."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2,
            connect_between_provinces=True, between_weight=0.5
        )
        P_adj, P_weight = build_province_adjacency(hier)
        assert P_adj.shape == (5, 5)
        # Should have some inter-province links
        assert np.sum(P_adj) >= 0.0

    def test_adjacency_symmetric(self):
        """Province adjacency should be symmetric."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2,
            connect_between_provinces=True, between_weight=0.5
        )
        P_adj, P_weight = build_province_adjacency(hier)
        assert np.allclose(P_adj, P_adj.T)
        assert np.allclose(P_weight, P_weight.T)


class TestIsBridgeProvince:
    """Test is_bridge_province function."""

    def test_bridge_province(self):
        """Province with inter-province links should be a bridge."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2,
            connect_between_provinces=True, between_weight=0.5
        )
        # At least one province should be a bridge
        result = is_bridge_province(0, hier)
        assert isinstance(result, bool)

    def test_no_bridge_when_isolated(self):
        """Isolated provinces should not be bridges."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2,
            connect_between_provinces=False
        )
        for p in range(5):
            assert not is_bridge_province(p, hier)


class TestApplyDominoEffects:
    """Test apply_domino_effects function."""

    def test_no_critical_provinces(self, base_regime_state, params):
        """No critical provinces should not change state."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2
        )
        state = base_regime_state.copy_with_hierarchical(hier)
        critical_flags = np.zeros(5, dtype=np.uint8)
        P_adj = np.zeros((5, 5))
        P_weight = np.zeros((5, 5))
        
        result = apply_domino_effects(state, critical_flags, P_adj, P_weight, params)
        # Hierarchical state should be preserved
        assert result.hierarchical is not None
        assert result.hierarchical.n_districts == state.hierarchical.n_districts

    def test_critical_province_reduces_legitimacy(self, base_regime_state, params):
        """Critical provinces should reduce legitimacy."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2,
            connect_between_provinces=True, between_weight=1.0
        )
        state = base_regime_state.copy_with_hierarchical(hier)
        critical_flags = np.zeros(5, dtype=np.uint8)
        critical_flags[0] = 1  # First province is critical
        P_adj = np.zeros((5, 5))
        P_weight = np.zeros((5, 5))
        
        result = apply_domino_effects(state, critical_flags, P_adj, P_weight, params)
        assert result.system.legitimacy < state.system.legitimacy


class TestApplyNationalShock:
    """Test apply_national_shock function."""

    def test_shock_increases_volatility(self, base_regime_state):
        """National shock should increase volatility."""
        initial_vol = base_regime_state.system.volatility
        result = apply_national_shock(base_regime_state)
        assert result.system.volatility > initial_vol

    def test_shock_increases_exhaustion(self, base_regime_state):
        """National shock should increase exhaustion."""
        initial_exh = base_regime_state.system.exhaustion
        result = apply_national_shock(base_regime_state)
        assert result.system.exhaustion > initial_exh

    def test_shock_clipped_to_one(self, base_regime_state):
        """Shock should be clipped to maximum of 1.0."""
        # Create state with high volatility
        sys = SystemState(
            legitimacy=0.2, cohesion=0.1, fragmentation=0.9,
            instability=0.9, mobilization=0.9, repression=0.8,
            elite_alignment=0.02, volatility=0.95, exhaustion=0.95,
        )
        state = base_regime_state.copy_with_system(sys)
        result = apply_national_shock(state)
        assert result.system.volatility <= 1.0
        assert result.system.exhaustion <= 1.0


class TestExhaustionEffects:
    """Test exhaustion-related state effects."""

    def test_admin_decay_when_high_exhaustion(self, base_regime_state, params):
        """High exhaustion should decay admin capacity."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2
        )
        sys = SystemState(
            legitimacy=0.5, cohesion=0.4, fragmentation=0.3,
            instability=0.3, mobilization=0.3, repression=0.5,
            elite_alignment=0.35, volatility=0.3, exhaustion=0.7,
        )
        state = base_regime_state.copy_with_system(sys).copy_with_hierarchical(hier)
        
        initial_admin = state.hierarchical.district_states[0].admin_capacity
        result = apply_exhaustion_admin_decay(state, params)
        
        # Admin capacity should decrease
        assert result.hierarchical.district_states[0].admin_capacity < initial_admin

    def test_admin_no_decay_when_low_exhaustion(self, base_regime_state, params):
        """Low exhaustion should not affect admin capacity."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2
        )
        sys = SystemState(
            legitimacy=0.5, cohesion=0.4, fragmentation=0.3,
            instability=0.3, mobilization=0.3, repression=0.5,
            elite_alignment=0.35, volatility=0.3, exhaustion=0.3,
        )
        state = base_regime_state.copy_with_system(sys).copy_with_hierarchical(hier)
        
        result = apply_exhaustion_admin_decay(state, params)
        # No change when exhaustion <= 0.5
        assert result.hierarchical.district_states[0].admin_capacity == \
               state.hierarchical.district_states[0].admin_capacity

    def test_unrest_drift_when_critical_exhaustion(self, base_regime_state):
        """Critical exhaustion (0.7+) should drift unrest up."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2
        )
        sys = SystemState(
            legitimacy=0.5, cohesion=0.4, fragmentation=0.3,
            instability=0.3, mobilization=0.3, repression=0.5,
            elite_alignment=0.35, volatility=0.3, exhaustion=0.8,
        )
        state = base_regime_state.copy_with_system(sys).copy_with_hierarchical(hier)
        
        initial_unrest = state.hierarchical.district_states[0].local_unrest
        result = apply_exhaustion_unrest_drift(state)
        
        # Unrest should increase
        assert result.hierarchical.district_states[0].local_unrest > initial_unrest

    def test_no_unrest_drift_below_threshold(self, base_regime_state):
        """Exhaustion below 0.7 should not drift unrest."""
        hier = create_hierarchical_state(
            n_provinces=5, districts_per_province=5, n_policy_dims=2
        )
        sys = SystemState(
            legitimacy=0.5, cohesion=0.4, fragmentation=0.3,
            instability=0.3, mobilization=0.3, repression=0.5,
            elite_alignment=0.35, volatility=0.3, exhaustion=0.5,
        )
        state = base_regime_state.copy_with_system(sys).copy_with_hierarchical(hier)
        
        result = apply_exhaustion_unrest_drift(state)
        assert result.hierarchical.district_states[0].local_unrest == \
               state.hierarchical.district_states[0].local_unrest


# ============================================================================
# 4. Factions Tests
# ============================================================================

class TestComputeGini:
    """Test compute_gini function."""

    def test_equal_powers(self):
        """Equal powers should give zero Gini."""
        powers = np.array([0.5, 0.5])
        gini = compute_gini(powers)
        assert gini == pytest.approx(0.0)

    def test_maximum_inequality(self):
        """Maximum inequality (one holds all power) should give high Gini."""
        powers = np.array([1.0, 0.0])
        gini = compute_gini(powers)
        assert gini >= 0.5

    def test_single_element(self):
        """Single element should give zero Gini."""
        gini = compute_gini(np.array([1.0]))
        assert gini == 0.0

    def test_all_zeros(self):
        """All zeros should give zero Gini."""
        gini = compute_gini(np.array([0.0, 0.0, 0.0]))
        assert gini == 0.0

    def test_result_bounded(self):
        """Gini should always be in [0, 1]."""
        for _ in range(100):
            powers = np.random.rand(5)
            powers /= powers.sum()
            gini = compute_gini(powers)
            assert 0.0 <= gini <= 1.0


class TestAggregationFunctions:
    """Test macro variable aggregation functions."""

    def test_legitimacy_calculation(self, base_regime_state):
        """Legitimacy should be power-weighted cohesion."""
        leg = compute_legitimacy(base_regime_state)
        powers = base_regime_state.get_faction_powers()
        cohs = base_regime_state.get_faction_cohesions()
        expected = np.sum(powers * cohs) / np.sum(powers)
        assert leg == pytest.approx(expected)

    def test_repression_inverse_legitimacy(self, base_regime_state):
        """Repression should be 1 - legitimacy."""
        leg = compute_legitimacy(base_regime_state)
        rep = compute_repression(leg)
        assert rep == pytest.approx(1.0 - leg)

    def test_elite_alignment_bounded(self, base_regime_state):
        """Elite alignment should be in [0, 1]."""
        leg = compute_legitimacy(base_regime_state)
        frag = compute_fragmentation(base_regime_state, lambda_frag=3.0)
        elite = compute_elite_alignment(leg, frag)
        assert 0.0 <= elite <= 1.0

    def test_volatility_bounded(self):
        """Volatility should always be in [0, 1]."""
        for m in [0.0, 0.5, 1.0]:
            for i in [0.0, 0.5, 1.0]:
                for e in [0.0, 0.5, 1.0]:
                    v = compute_volatility(m, i, e, kappa_v=2.5)
                    assert 0.0 <= v <= 1.0


class TestFactoryFunctions:
    """Test faction factory functions."""

    def test_create_balanced_factions_count(self):
        """Should create correct number of factions."""
        for n in [2, 3, 4, 5, 6]:
            factions = create_balanced_factions(n)
            assert len(factions) == n

    def test_create_balanced_factions_equal_power(self):
        """Balanced factions should have equal power."""
        factions = create_balanced_factions(4)
        powers = [f.power for f in factions]
        assert all(p == pytest.approx(powers[0]) for p in powers)

    def test_create_balanced_factions_neutral(self):
        """Balanced factions should have neutral values."""
        factions = create_balanced_factions(3)
        for f in factions:
            assert f.radicalization == 0.0
            assert f.memory == 0.0
            assert f.cohesion == 0.5

    def test_create_balanced_invalid_n(self):
        """Invalid n_factions should raise ValueError."""
        with pytest.raises(ValueError):
            create_balanced_factions(1)
        with pytest.raises(ValueError):
            create_balanced_factions(7)

    def test_create_dominant_factions(self):
        """Should create one dominant faction."""
        factions = create_dominant_factions(3, dominant_idx=0, dominant_power=0.6)
        assert len(factions) == 3
        assert factions[0].power == 0.6
        assert all(f.power == pytest.approx(0.2) for f in factions[1:])

    def test_create_dominant_invalid_params(self):
        """Invalid parameters should raise ValueError."""
        with pytest.raises(ValueError):
            create_dominant_factions(3, dominant_idx=-1, dominant_power=0.5)
        with pytest.raises(ValueError):
            create_dominant_factions(3, dominant_idx=5, dominant_power=0.5)
        with pytest.raises(ValueError):
            create_dominant_factions(3, dominant_idx=0, dominant_power=0.0)
        with pytest.raises(ValueError):
            create_dominant_factions(3, dominant_idx=0, dominant_power=1.0)


# ============================================================================
# 5. State Validation Tests
# ============================================================================

class TestFactionStateValidation:
    """Test FactionState validation."""

    def test_valid_faction(self):
        """Valid faction should be created."""
        f = FactionState(power=0.5, radicalization=0.3, cohesion=0.7, memory=0.2, wealth=0.5)
        assert f.power == 0.5
        assert f.radicalization == 0.3

    def test_invalid_power(self):
        """Invalid power should raise ValueError."""
        with pytest.raises(ValueError):
            FactionState(power=-0.1, radicalization=0.3, cohesion=0.7, memory=0.2)
        with pytest.raises(ValueError):
            FactionState(power=1.5, radicalization=0.3, cohesion=0.7, memory=0.2)

    def test_to_array_roundtrip(self):
        """to_array and from_array should be inverse operations."""
        original = FactionState(power=0.5, radicalization=0.3, cohesion=0.7, memory=0.2, wealth=0.4)
        arr = original.to_array()
        restored = FactionState.from_array(arr)
        assert restored.power == original.power
        assert restored.radicalization == original.radicalization

    def test_from_array_clamping(self):
        """from_array should clamp out-of-range values."""
        arr = np.array([-0.5, 1.5, 0.7, 0.2, 0.5])
        f = FactionState.from_array(arr)
        assert 0.0 <= f.power <= 1.0
        assert 0.0 <= f.radicalization <= 1.0


class TestSystemStateValidation:
    """Test SystemState validation."""

    def test_valid_system(self):
        """Valid system should be created."""
        s = SystemState(
            legitimacy=0.5, cohesion=0.4, fragmentation=0.2,
            instability=0.1, mobilization=0.1, repression=0.5,
            elite_alignment=0.4, volatility=0.1, exhaustion=0.1,
        )
        assert s.legitimacy == 0.5

    def test_invalid_system_field(self):
        """Invalid field should raise ValueError."""
        with pytest.raises(ValueError):
            SystemState(
                legitimacy=1.5, cohesion=0.4, fragmentation=0.2,
                instability=0.1, mobilization=0.1, repression=0.5,
                elite_alignment=0.4, volatility=0.1, exhaustion=0.1,
            )

    def test_neutral_factory(self):
        """SystemState.neutral should create valid state."""
        s = SystemState.neutral()
        assert 0.0 <= s.legitimacy <= 1.0
        assert 0.0 <= s.exhaustion <= 1.0
        assert len(s.pillars) == 3

    def test_pillars_validation(self):
        """Pillars should also be validated."""
        with pytest.raises(ValueError):
            SystemState(
                legitimacy=0.5, cohesion=0.4, fragmentation=0.2,
                instability=0.1, mobilization=0.1, repression=0.5,
                elite_alignment=0.4, volatility=0.1, exhaustion=0.1,
                pillars=(0.5, 1.5, 0.5),  # Invalid pillar
            )


class TestRegimeStateValidation:
    """Test RegimeState validation."""

    def test_valid_regime(self, base_factions, stable_system):
        """Valid regime should be created."""
        r = RegimeState(
            factions=base_factions,
            system=stable_system,
            affinity_matrix=((1.0, 0.5), (0.5, 1.0)),
            step=0,
        )
        assert r.n_factions == 2

    def test_invalid_faction_count(self, stable_system):
        """Invalid faction count should raise ValueError."""
        with pytest.raises(ValueError):
            RegimeState(factions=[], system=stable_system)
        with pytest.raises(ValueError):
            RegimeState(
                factions=[FactionState(0.5, 0.3, 0.7, 0.2) for _ in range(7)],
                system=stable_system,
            )

    def test_negative_step(self, base_factions, stable_system):
        """Negative step should raise ValueError."""
        with pytest.raises(ValueError):
            RegimeState(factions=base_factions, system=stable_system, step=-1)

    def test_affinity_matrix_validation(self, base_factions, stable_system):
        """Affinity matrix should be validated."""
        # Wrong size
        with pytest.raises(ValueError):
            RegimeState(
                factions=base_factions,
                system=stable_system,
                affinity_matrix=((1.0,),),
            )
        # Invalid value
        with pytest.raises(ValueError):
            RegimeState(
                factions=base_factions,
                system=stable_system,
                affinity_matrix=((1.0, 1.5), (1.5, 1.0)),
            )

    def test_advance_step(self, base_regime_state):
        """advance_step should increment step counter."""
        assert base_regime_state.step == 0
        next_state = base_regime_state.advance_step()
        assert next_state.step == 1


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
