"""
Michigan Senate Democratic Primary (El-Sayed vs Stevens)
Live election-night Bayesian county-level model

Scalable framework for real-time margin projection with credibility-weighted
statewide shift detection and outlier dampening.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Tuple, List
import json


# ============================================================================
# BASELINE DATA SETUP
# ============================================================================

# Michigan regional structure (geographic + demographic clustering)
COUNTY_REGIONS = {
    # Detroit Metro (Southeast) - urban/diverse
    'Wayne': 'Detroit_Metro',
    'Oakland': 'Detroit_Metro',
    'Macomb': 'Detroit_Metro',
    'St. Clair': 'Detroit_Metro',
    'Monroe': 'Detroit_Metro',
    'Lenawee': 'Detroit_Metro',
    'Jackson': 'Detroit_Metro',
    'Hillsdale': 'Detroit_Metro',
    
    # Mid-Michigan (Central) - mixed urban/rural, Lansing area
    'Ingham': 'Mid_Michigan',
    'Eaton': 'Mid_Michigan',
    'Clinton': 'Mid_Michigan',
    'Shiawassee': 'Mid_Michigan',
    'Genesee': 'Mid_Michigan',
    'Saginaw': 'Mid_Michigan',
    'Gratiot': 'Mid_Michigan',
    'Bay': 'Mid_Michigan',
    
    # West Michigan - Grand Rapids area, coast
    'Kent': 'West_Michigan',
    'Ottawa': 'West_Michigan',
    'Allegan': 'West_Michigan',
    'Barry': 'West_Michigan',
    'Montcalm': 'West_Michigan',
    'Ionia': 'West_Michigan',
    'Newaygo': 'West_Michigan',
    'Muskegon': 'West_Michigan',
    'Kalamazoo': 'West_Michigan',
    'Van Buren': 'West_Michigan',
    'Cass': 'West_Michigan',
    'Berrien': 'West_Michigan',
    'Calhoun': 'West_Michigan',
    'Branch': 'West_Michigan',
    'Mecosta': 'West_Michigan',
    
    # Thumb Region - rural agricultural
    'Tuscola': 'Thumb',
    'Huron': 'Thumb',
    'Sanilac': 'Thumb',
    'Lapeer': 'Thumb',
    
    # Northern Lower Peninsula - rural/small towns, resort areas
    'Grand Traverse': 'North_Lower',
    'Leelanau': 'North_Lower',
    'Charlevoix': 'North_Lower',
    'Emmet': 'North_Lower',
    'Antrim': 'North_Lower',
    'Otsego': 'North_Lower',
    'Wexford': 'North_Lower',
    'Missaukee': 'North_Lower',
    'Kalkaska': 'North_Lower',
    'Manistee': 'North_Lower',
    'Benzie': 'North_Lower',
    'Mason': 'North_Lower',
    'Lake': 'North_Lower',
    'Osceola': 'North_Lower',
    'Clare': 'North_Lower',
    'Isabella': 'North_Lower',
    'Midland': 'North_Lower',
    'Gladwin': 'North_Lower',
    'Roscommon': 'North_Lower',
    'Ogemaw': 'North_Lower',
    'Iosco': 'North_Lower',
    'Oscoda': 'North_Lower',
    'Arenac': 'North_Lower',
    'Alcona': 'North_Lower',
    'Montmorency': 'North_Lower',
    
    # Upper Peninsula - rural, mining heritage
    'Marquette': 'Upper_Peninsula',
    'Houghton': 'Upper_Peninsula',
    'Delta': 'Upper_Peninsula',
    'Menominee': 'Upper_Peninsula',
    'Dickinson': 'Upper_Peninsula',
    'Iron': 'Upper_Peninsula',
    'Gogebic': 'Upper_Peninsula',
    'Ontonagon': 'Upper_Peninsula',
    'Baraga': 'Upper_Peninsula',
    'Alger': 'Upper_Peninsula',
    'Schoolcraft': 'Upper_Peninsula',
    'Luce': 'Upper_Peninsula',
    'Mackinac': 'Upper_Peninsula',
    'Chippewa': 'Upper_Peninsula',
    'Presque Isle': 'Upper_Peninsula',
    'Alpena': 'Upper_Peninsula',
}

# County-level baseline confidence
# High-confidence: Wayne, Macomb, Oakland, Kent, Washtenaw, Genesee (strong polling/historical data)
# Medium-confidence: Most other counties (~20)
# Low-confidence: Small rural counties (~55)
# This affects how much the Bayesian shift moves each county's projection

COUNTY_BASELINE_CONFIDENCE = {
    # High confidence (1.0x) - strong polling/historical data
    'Wayne': 1.0,
    'Macomb': 1.0,
    'Oakland': 1.0,
    'Kent': 1.0,
    'Washtenaw': 1.0,
    'Genesee': 1.0,
    
    # Medium confidence (0.6x) - moderate data, larger metro/county
    'Ingham': 0.6,
    'Kalamazoo': 0.6,
    'Saginaw': 0.6,
    'Eaton': 0.6,
    'Ottawa': 0.6,
    'Calhoun': 0.6,
    'Jackson': 0.6,
    'Berrien': 0.6,
    'Grand Traverse': 0.6,
    'Midland': 0.6,
    'Bay': 0.6,
    'Monroe': 0.6,
    'St. Clair': 0.6,
    'Allegan': 0.6,
    
    # Low confidence (0.25x) - sparse data, small/rural counties
    'St. Joseph': 0.25, 'Tuscola': 0.25, 'Montcalm': 0.25, 'Ionia': 0.25, 'Newaygo': 0.25,
    'Cass': 0.25, 'Sanilac': 0.25, 'Delta': 0.25, 'Emmet': 0.25, 'Hillsdale': 0.25,
    'Branch': 0.25, 'Houghton': 0.25, 'Mecosta': 0.25, 'Charlevoix': 0.25, 'Gratiot': 0.25,
    'Wexford': 0.25, 'Chippewa': 0.25, 'Huron': 0.25, 'Antrim': 0.25, 'Mason': 0.25,
    'Alpena': 0.25, 'Cheboygan': 0.25, 'Dickinson': 0.25, 'Leelanau': 0.25, 'Otsego': 0.25,
    'Manistee': 0.25, 'Clare': 0.25, 'Iosco': 0.25, 'Menominee': 0.25, 'Gladwin': 0.25,
    'Roscommon': 0.25, 'Oceana': 0.25, 'Ogemaw': 0.25, 'Osceola': 0.25, 'Benzie': 0.25,
    'Kalkaska': 0.25, 'Arenac': 0.25, 'Missaukee': 0.25, 'Crawford': 0.25, 'Gogebic': 0.25,
    'Presque Isle': 0.25, 'Mackinac': 0.25, 'Iron': 0.25, 'Montmorency': 0.25, 'Alcona': 0.25,
    'Lake': 0.25, 'Alger': 0.25, 'Schoolcraft': 0.25, 'Baraga': 0.25, 'Oscoda': 0.25,
    'Ontonagon': 0.25, 'Luce': 0.25, 'Keweenaw': 0.25, 'Lapeer': 0.25, 'Livingston': 0.25,
    'Clinton': 0.25, 'Shiawassee': 0.25, 'Marquette': 0.25, 'Van Buren': 0.25, 'Isabella': 0.25,
    'Lenawee': 0.25, 'Barry': 0.25, 'Muskegon': 0.25,
}

# County-level baseline margins (El-Sayed margin in points)
COUNTY_BASELINES = {
    'Wayne': 8.0, 'Oakland': 8.3, 'Macomb': 12.1, 'Kent': 30.9, 'Washtenaw': 39.4,
    'Genesee': 16.5, 'Ottawa': 41.5, 'Kalamazoo': 21.9, 'Ingham': 53.7, 'Livingston': 51.4,
    'Saginaw': 4.6, 'Muskegon': 12.0, 'St. Clair': 4.3, 'Berrien': 5.2, 'Monroe': 24.5,
    'Jackson': 1.4, 'Allegan': 15.2, 'Calhoun': 10.9, 'Bay': 4.0, 'Eaton': 33.2,
    'Grand Traverse': 37.4, 'Lapeer': 1.1, 'Midland': 33.7, 'Lenawee': 4.5, 'Clinton': 37.9,
    'Marquette': 31.4, 'Shiawassee': 14.5, 'Barry': 11.7, 'Van Buren': 19.2, 'Isabella': 39.5,
    'St. Joseph': -2.2, 'Tuscola': 3.8, 'Montcalm': 8.8, 'Ionia': 16.0, 'Newaygo': 19.5,
    'Cass': 1.1, 'Sanilac': 1.0, 'Delta': 0.0, 'Emmet': 29.9, 'Hillsdale': 3.5,
    'Branch': 9.6, 'Houghton': 35.7, 'Mecosta': 10.0, 'Charlevoix': 38.2, 'Gratiot': -4.0,
    'Wexford': 0.1, 'Chippewa': 1.1, 'Huron': 9.2, 'Antrim': 15.0, 'Mason': 9.9,
    'Alpena': 8.3, 'Cheboygan': 11.1, 'Dickinson': 9.2, 'Leelanau': 66.5, 'Otsego': -6.8,
    'Manistee': 20.0, 'Clare': 4.3, 'Iosco': 8.6, 'Menominee': 2.1, 'Gladwin': 22.8,
    'Roscommon': 4.9, 'Oceana': -0.5, 'Ogemaw': 21.3, 'Osceola': 1.6, 'Benzie': 36.1,
    'Kalkaska': 5.4, 'Arenac': 6.0, 'Missaukee': 3.9, 'Crawford': 17.0, 'Gogebic': 4.4,
    'Presque Isle': 8.0, 'Mackinac': 0.4, 'Iron': 3.7, 'Montmorency': 14.6, 'Alcona': 11.6,
    'Lake': -5.9, 'Alger': 0.1, 'Schoolcraft': 12.8, 'Baraga': 2.3, 'Oscoda': 9.4,
    'Ontonagon': -0.3, 'Luce': 1.6, 'Keweenaw': 5.9
}

# Historical Slotkin-Harper 2024 turnout (used as distribution template)
SLOTKIN_HARPER_TURNOUT = {
    'Wayne': 194835, 'Oakland': 152510, 'Macomb': 70816, 'Washtenaw': 62639,
    'Kent': 62509, 'Genesee': 43731, 'Ingham': 36376, 'Kalamazoo': 26401,
    'Saginaw': 19758, 'Livingston': 16633, 'Muskegon': 16098, 'Eaton': 11208,
    'Grand Traverse': 10880, 'Bay': 10388, 'Ottawa': 10259, 'Jackson': 9443,
    'St. Clair': 9328, 'Monroe': 9290, 'Berrien': 9261, 'Marquette': 9079,
    'Clinton': 7742, 'Calhoun': 7517, 'Allegan': 7113, 'Midland': 6981,
    'Lenawee': 5391, 'Shiawassee': 5279, 'Lapeer': 5152, 'Van Buren': 4976,
    'Isabella': 4923, 'Leelanau': 4289, 'Montcalm': 3172, 'Emmet': 3080,
    'Benzie': 2721, 'Ionia': 2721, 'Alpena': 2691, 'Houghton': 2685,
    'Delta': 2596, 'Charlevoix': 2530, 'Tuscola': 2483, 'Manistee': 2381,
    'Chippewa': 2154, 'Cass': 2135, 'Mecosta': 1967, 'Iosco': 1958,
    'Newaygo': 1945, 'Barry': 1913, 'Antrim': 1895, 'Roscommon': 1889,
    'Mason': 1808, 'Cheboygan': 1799, 'Dickinson': 1727, 'Sanilac': 1712,
    'Otsego': 1694, 'Clare': 1684, 'Gratiot': 1643, 'Oceana': 1611,
    'Wexford': 1562, 'Menominee': 1527, 'Hillsdale': 1523, 'Huron': 1515,
    'Gladwin': 1511, 'St. Joseph': 1509, 'Ogemaw': 1405, 'Branch': 1344,
    'Presque Isle': 1229, 'Gogebic': 1176, 'Kalkaska': 1136, 'Iron': 1028,
    'Alger': 977, 'Osceola': 951, 'Crawford': 903, 'Mackinac': 877,
    'Arenac': 824, 'Alcona': 813, 'Lake': 811, 'Ontonagon': 636,
    'Missaukee': 605, 'Baraga': 575, 'Schoolcraft': 554, 'Montmorency': 546,
    'Oscoda': 391, 'Keweenaw': 277, 'Luce': 240
}

TARGET_TURNOUT = 1_200_000
SLOTKIN_HARPER_TOTAL = sum(SLOTKIN_HARPER_TURNOUT.values())
TURNOUT_SCALE_FACTOR = TARGET_TURNOUT / SLOTKIN_HARPER_TOTAL

# Scale all counties proportionally
BASELINE_TURNOUT = {
    county: int(votes * TURNOUT_SCALE_FACTOR)
    for county, votes in SLOTKIN_HARPER_TURNOUT.items()
}

# McMorrow vote share will be calculated dynamically from observed votes
# (no longer hardcoded)

# Model hyperparameters
@dataclass
class ModelConfig:
    """Bayesian model configuration"""
    # DerSimonian-Laird meta-analysis
    credibility_exponent: float = 2.0  # Controls how quickly shift confidence grows
    tau_floor: float = 0.08  # Minimum heterogeneity (primary uncertainty buffer)
    
    # Outlier dampening (keep weird counties from overriding the model)
    outlier_lambda: float = 3.0  # Robustness scaling—higher = more aggressive dampening
    outlier_threshold_z: float = 2.5  # Z-score threshold for flagging outliers
    
    # Deductive reasoning constraint: momentum (early vote ≠ final by >X points)
    momentum_constraint_threshold: float = 0.30  # Pct reported to activate (30%+)
    momentum_max_drift: float = 10.0  # Max margin drift from observed (points)
    
    # Simulation
    n_simulations: int = 20_000
    
    # Projection sensitivity
    adaptation_speed: float = 1.0  # 1.0 = full Bayesian update, <1 = slower


# ============================================================================
# BASELINE VOTE PROJECTION
# ============================================================================

class BaselineProjection:
    """Calculate expected vote totals from baselines and turnout"""
    
    def __init__(
        self,
        baselines: Dict[str, float],
        turnout: Dict[str, int],
        confidence: Dict[str, float] = None
    ):
        self.baselines = baselines
        self.turnout = turnout
        self.confidence = confidence or {c: 1.0 for c in baselines.keys()}
        self.projection = self._compute_baseline()
    
    def _compute_baseline(self) -> pd.DataFrame:
        """Build county-level baseline projections"""
        counties = []
        
        for county in self.baselines.keys():
            margin = self.baselines[county]
            votes = self.turnout[county]
            
            # Convert margin to vote shares
            # If El-Sayed margin is +8, El-Sayed gets 54%, Stevens 44%
            # McMorrow share will be calculated from observed votes dynamically
            el_sayed_share = (margin + 100) / 2 / 100
            stevens_share = (100 - margin) / 2 / 100
            
            # Project El-Sayed vs Stevens (McMorrow calculated from observed data)
            el_sayed_votes = int(votes * el_sayed_share)
            stevens_votes = int(votes * stevens_share)
            
            counties.append({
                'county': county,
                'baseline_margin': margin,
                'total_votes': votes,
                'el_sayed_baseline': el_sayed_votes,
                'stevens_baseline': stevens_votes,
            })
        
        return pd.DataFrame(counties)
    
    def get_statewide_baseline(self) -> Dict[str, float]:
        """Calculate statewide baseline (El-Sayed vs Stevens only)"""
        total_el_sayed = self.projection['el_sayed_baseline'].sum()
        total_stevens = self.projection['stevens_baseline'].sum()
        total_votes = total_el_sayed + total_stevens
        
        margin = (total_el_sayed - total_stevens) / total_votes * 100
        
        return {
            'el_sayed': total_el_sayed,
            'stevens': total_stevens,
            'total': total_votes,
            'el_sayed_pct': total_el_sayed / total_votes,
            'stevens_pct': total_stevens / total_votes,
            'margin': margin
        }


# ============================================================================
# LIVE VOTE PROCESSING
# ============================================================================

class LiveVoteAggregator:
    """Track incoming vote totals and compute observed margins"""
    
    def __init__(self, baseline_projection: BaselineProjection):
        self.baseline = baseline_projection
        self.observed_votes = {}
    
    def add_county_results(self, county: str, el_sayed: int, stevens: int, mcmorrow: int = 0):
        """Add observed votes for a county"""
        self.observed_votes[county] = {
            'el_sayed': el_sayed,
            'stevens': stevens,
            'mcmorrow': mcmorrow,
            'total': el_sayed + stevens + mcmorrow
        }
    
    def get_observed_margins(self) -> pd.DataFrame:
        """Compute observed margin vs baseline for all reported counties"""
        rows = []
        
        for county, votes in self.observed_votes.items():
            if county not in self.baseline.baselines:
                continue
            
            baseline_margin = self.baseline.baselines[county]
            observed_margin = (votes['el_sayed'] - votes['stevens']) / (votes['el_sayed'] + votes['stevens']) * 100
            deviation = observed_margin - baseline_margin
            pct_reported = votes['total'] / self.baseline.turnout[county]
            
            rows.append({
                'county': county,
                'baseline_margin': baseline_margin,
                'observed_margin': observed_margin,
                'deviation': deviation,
                'votes_in': votes['total'],
                'pct_reported': pct_reported
            })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ============================================================================
# BAYESIAN SHIFT ESTIMATION
# ============================================================================

class BayesianShiftEstimator:
    """Detect and quantify statewide shift using DerSimonian-Laird meta-analysis"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
    
    def estimate_statewide_shift(self, observed_margins_df: pd.DataFrame) -> Dict:
        """
        Meta-analyze county deviations to estimate true statewide shift.
        
        Uses inverse-variance weighting with credibility adjustment:
        - Counties with more votes in → higher weight
        - Heterogeneity tau controls robustness to outliers
        - Credibility interval widens as votes trickle in
        """
        
        if len(observed_margins_df) == 0:
            return {
                'statewide_shift': 0.0,
                'se_shift': float('inf'),
                'ci_lower': -float('inf'),
                'ci_upper': float('inf'),
                'n_counties': 0,
                'heterogeneity_tau': 0.0,
                'effective_weight': 0.0,
                'regional_shifts': {}
            }
        
        df = observed_margins_df.copy()
        
        # Weight by percentage of votes reported (more votes = more certain estimate)
        # Use squared percentage to heavily penalize partial reporting
        df['weight'] = df['pct_reported'] ** 2
        
        # Compute individual variances
        # Assume binomial: var ≈ 4 * p * (1-p) / n, simplified to county variance
        df['variance'] = 1.0 / (df['weight'] + 0.01)  # Avoid division by zero
        
        # DerSimonian-Laird heterogeneity estimation
        # Compute Q statistic
        weighted_mean = (df['deviation'] * df['weight']).sum() / df['weight'].sum()
        Q = (df['weight'] * (df['deviation'] - weighted_mean) ** 2).sum()
        k = len(df)
        W = df['weight'].sum()
        W2 = (df['weight'] ** 2).sum()
        
        # Heterogeneity variance (tau^2)
        tau_sq = max(
            (Q - (k - 1)) / (W - W2 / W),
            self.config.tau_floor ** 2  # Primary uncertainty buffer
        )
        tau = np.sqrt(tau_sq)
        
        # Adjusted weights incorporating heterogeneity
        df['adjusted_variance'] = df['variance'] + tau_sq
        df['adjusted_weight'] = 1.0 / df['adjusted_variance']
        
        # Outlier dampening: Robustify against high-deviation counties
        standardized_dev = (df['deviation'] - weighted_mean) / (tau + 0.001)
        outlier_factor = 1.0 / (1.0 + (np.abs(standardized_dev) / self.config.outlier_lambda) ** 2)
        df['outlier_dampened_weight'] = df['adjusted_weight'] * outlier_factor
        
        # Final weighted mean
        final_shift = (df['deviation'] * df['outlier_dampened_weight']).sum() / df['outlier_dampened_weight'].sum()
        se_shift = np.sqrt(1.0 / df['outlier_dampened_weight'].sum())
        
        # Credibility interval based on available data
        z_crit = 1.96
        ci_lower = final_shift - z_crit * se_shift
        ci_upper = final_shift + z_crit * se_shift
        
        # Effective weight: measure of how much data is driving the estimate
        effective_weight = df['outlier_dampened_weight'].sum() / df['weight'].max()
        
        # Compute regional shifts
        regional_shifts = self._estimate_regional_shifts(df)
        
        return {
            'statewide_shift': final_shift,
            'se_shift': se_shift,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_counties': len(df),
            'heterogeneity_tau': tau,
            'effective_weight': min(effective_weight, 1.0),  # Cap at 1.0
            'details_df': df,  # For debugging/transparency
            'regional_shifts': regional_shifts
        }
    
    def _estimate_regional_shifts(self, observed_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Compute DerSimonian-Laird shift estimate for each region separately.
        
        Returns dict keyed by region with shift, SE, CI, n_counties
        """
        
        regional_shifts = {}
        
        # Get unique regions from observed counties
        regions_in_data = set()
        for county in observed_df['county']:
            if county in COUNTY_REGIONS:
                regions_in_data.add(COUNTY_REGIONS[county])
        
        for region in regions_in_data:
            region_data = observed_df[observed_df['county'].map(
                lambda c: COUNTY_REGIONS.get(c) == region
            )]
            
            if len(region_data) == 0:
                continue
            
            # Mini DerSimonian-Laird for this region
            df = region_data.copy()
            
            weighted_mean = (df['deviation'] * df['weight']).sum() / df['weight'].sum()
            Q = (df['weight'] * (df['deviation'] - weighted_mean) ** 2).sum()
            k = len(df)
            W = df['weight'].sum()
            W2 = (df['weight'] ** 2).sum()
            
            tau_sq = max(
                (Q - (k - 1)) / (W - W2 / W) if k > 1 else 0,
                (self.config.tau_floor / 2) ** 2  # Slightly looser for regional
            )
            tau = np.sqrt(tau_sq)
            
            df['adjusted_variance'] = df['variance'] + tau_sq
            df['adjusted_weight'] = 1.0 / df['adjusted_variance']
            
            standardized_dev = (df['deviation'] - weighted_mean) / (tau + 0.001)
            outlier_factor = 1.0 / (1.0 + (np.abs(standardized_dev) / self.config.outlier_lambda) ** 2)
            df['outlier_dampened_weight'] = df['adjusted_weight'] * outlier_factor
            
            regional_shift = (df['deviation'] * df['outlier_dampened_weight']).sum() / df['outlier_dampened_weight'].sum()
            se_regional = np.sqrt(1.0 / df['outlier_dampened_weight'].sum())
            
            z_crit = 1.96
            ci_lower = regional_shift - z_crit * se_regional
            ci_upper = regional_shift + z_crit * se_regional
            
            regional_shifts[region] = {
                'shift': regional_shift,
                'se': se_regional,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_counties': len(df),
                'tau': tau
            }
        
        return regional_shifts


# ============================================================================
# MOMENTUM CONSTRAINT (Deductive Reasoning)
# ============================================================================

class MomentumConstrainer:
    """
    Enforce consistency: if significant votes are in, final margin can't drift
    more than ~10 points from observed.
    
    Deductive logic: early vote margin is predictive of final margin.
    If county runs El-Sayed +15 on 40% of votes, the final can't be +25 or +5.
    This constrains the unreported portion's implied margin.
    """
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
    
    def apply_constraint(
        self,
        county: str,
        observed_margin: float,
        pct_reported: float,
        projected_margin: float
    ) -> Tuple[float, Dict]:
        """
        Apply momentum constraint to a county's projected margin.
        
        Args:
            county: County name
            observed_margin: Margin from votes currently reported
            pct_reported: Fraction of county votes currently in (0-1)
            projected_margin: Bayesian projection before constraint
        
        Returns:
            constrained_margin: Final margin after applying constraint
            constraint_info: Debug info on whether constraint was active
        """
        
        # Only apply if sufficient votes reported
        if pct_reported < self.config.momentum_constraint_threshold:
            return projected_margin, {
                'constraint_active': False,
                'reason': f'Only {pct_reported:.1%} reported; threshold is {self.config.momentum_constraint_threshold:.1%}',
                'observed': observed_margin,
                'projected_before': projected_margin,
                'projected_after': projected_margin,
                'drift': 0.0
            }
        
        # Compute drift: how far is projection from observed?
        drift = projected_margin - observed_margin
        max_allowed_drift = self.config.momentum_max_drift
        
        # If drift exceeds threshold, clamp it
        if abs(drift) > max_allowed_drift:
            # Clamp drift to ±max_allowed_drift
            clamped_drift = np.clip(drift, -max_allowed_drift, max_allowed_drift)
            constrained_margin = observed_margin + clamped_drift
            
            return constrained_margin, {
                'constraint_active': True,
                'reason': f'Drift of {drift:+.1f} exceeds ±{max_allowed_drift:.1f} point limit',
                'observed': observed_margin,
                'projected_before': projected_margin,
                'projected_after': constrained_margin,
                'drift': clamped_drift,
                'drift_reduced_by': drift - clamped_drift
            }
        else:
            # No constraint needed
            return projected_margin, {
                'constraint_active': False,
                'reason': f'Drift of {drift:+.1f} is within ±{max_allowed_drift:.1f} point limit',
                'observed': observed_margin,
                'projected_before': projected_margin,
                'projected_after': projected_margin,
                'drift': drift
            }


# ============================================================================
# COUNTY-LEVEL PROJECTION
# ============================================================================

class CountyProjector:
    """Adjust county baselines based on statewide shift, with momentum constraint"""
    
    def __init__(
        self,
        baseline_projection: BaselineProjection,
        vote_aggregator: 'LiveVoteAggregator' = None,
        config: ModelConfig = None
    ):
        self.baseline = baseline_projection
        self.vote_aggregator = vote_aggregator
        self.config = config or ModelConfig()
        self.momentum = MomentumConstrainer(self.config)
    
    def project_county(self, county: str, statewide_shift_estimate: Dict, mcmorrow_share: float = None) -> Dict:
        """
        Project county margin and votes given statewide shift.
        
        Uses regional shift if available, otherwise falls back to global.
        Low-confidence counties (sparse polling) move less from baseline and have wider CIs.
        
        Adjustment = baseline + (confidence_adjusted credibility_weighted_shift)
        Credibility weight grows with data in and statewide shift confidence.
        
        If significant votes reported, applies momentum constraint:
        final margin can't drift >10 points from observed.
        """
        
        # Get baseline confidence for this county
        confidence = self.baseline.confidence.get(county, 1.0)
        
        # Determine which shift to use
        region = COUNTY_REGIONS.get(county)
        regional_shifts = statewide_shift_estimate.get('regional_shifts', {})
        
        # Prefer regional shift if available, otherwise use global
        if region and region in regional_shifts and regional_shifts[region]['n_counties'] > 0:
            shift = regional_shifts[region]['shift']
        else:
            shift = statewide_shift_estimate['statewide_shift']
        
        credibility = statewide_shift_estimate['effective_weight'] ** self.config.credibility_exponent
        
        # Apply confidence adjustment: low-confidence counties move less from baseline
        # High confidence (1.0): Full shift applied
        # Medium confidence (0.6): 60% of shift applied
        # Low confidence (0.25): 25% of shift applied
        confidence_adjusted_shift = shift * confidence
        
        # Bayesian adjustment (confidence-modulated)
        adjusted_margin = self.baseline.baselines[county] + (
            confidence_adjusted_shift * credibility * self.config.adaptation_speed
        )
        
        # Apply momentum constraint if we have observed votes for this county
        momentum_info = {'constraint_active': False}
        if self.vote_aggregator and county in self.vote_aggregator.observed_votes:
            votes_data = self.vote_aggregator.observed_votes[county]
            observed_margin = (votes_data['el_sayed'] - votes_data['stevens']) / (votes_data['el_sayed'] + votes_data['stevens']) * 100
            pct_reported = votes_data['total'] / self.baseline.turnout[county]
            
            adjusted_margin, momentum_info = self.momentum.apply_constraint(
                county, observed_margin, pct_reported, adjusted_margin
            )
        
        # Recalculate votes based on final adjusted margin
        votes = self.baseline.turnout[county]
        
        # Use observed McMorrow share (default 0 if no observed data)
        observed_mcmorrow = mcmorrow_share or 0.0
        mcmorrow_votes = int(votes * observed_mcmorrow)
        remaining_votes = votes - mcmorrow_votes
        
        el_sayed_share = (adjusted_margin + 100) / 2 / 100
        stevens_share = (100 - adjusted_margin) / 2 / 100
        
        el_sayed_votes = int(remaining_votes * el_sayed_share)
        stevens_votes = remaining_votes - el_sayed_votes
        
        return {
            'county': county,
            'region': region,
            'baseline_margin': self.baseline.baselines[county],
            'baseline_confidence': confidence,
            'adjusted_margin': adjusted_margin,
            'shift_applied': confidence_adjusted_shift * credibility,
            'shift_source': 'regional' if region and region in regional_shifts and regional_shifts[region]['n_counties'] > 0 else 'statewide',
            'el_sayed_projected': el_sayed_votes,
            'stevens_projected': stevens_votes,
            'mcmorrow_projected': mcmorrow_votes,
            'total_projected': votes,
            'momentum_constraint': momentum_info
        }
    
    def project_all_counties(self, statewide_shift_estimate: Dict) -> pd.DataFrame:
        """Project all counties given statewide shift"""
        
        # Calculate observed McMorrow share from reported votes
        mcmorrow_share = self._calculate_mcmorrow_share()
        
        projections = []
        
        for county in self.baseline.baselines.keys():
            proj = self.project_county(county, statewide_shift_estimate, mcmorrow_share)
            projections.append(proj)
        
        return pd.DataFrame(projections)
    
    def _calculate_mcmorrow_share(self) -> float:
        """Calculate McMorrow's observed share from reported votes"""
        if not self.vote_aggregator or not self.vote_aggregator.observed_votes:
            return 0.0
        
        total_mcmorrow = 0
        total_votes = 0
        
        for votes_data in self.vote_aggregator.observed_votes.values():
            total_mcmorrow += votes_data.get('mcmorrow', 0)
            total_votes += votes_data['total']
        
        if total_votes == 0:
            return 0.0
        
        return total_mcmorrow / total_votes


# ============================================================================
# STATEWIDE AGGREGATION & SIMULATION
# ============================================================================

class StatewideProjector:
    """Aggregate county projections to statewide and generate confidence intervals"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
    
    def aggregate_statewide(self, county_projections_df: pd.DataFrame) -> Dict:
        """Sum county-level projections to statewide"""
        
        total_el_sayed = county_projections_df['el_sayed_projected'].sum()
        total_stevens = county_projections_df['stevens_projected'].sum()
        total_mcmorrow = county_projections_df['mcmorrow_projected'].sum()
        total_votes = total_el_sayed + total_stevens + total_mcmorrow
        
        if total_votes == 0:
            return {
                'el_sayed': 0,
                'stevens': 0,
                'mcmorrow': 0,
                'total': 0,
                'el_sayed_pct': 0.0,
                'stevens_pct': 0.0,
                'mcmorrow_pct': 0.0,
                'margin': 0.0
            }
        
        margin = (total_el_sayed - total_stevens) / (total_el_sayed + total_stevens) * 100
        
        return {
            'el_sayed': total_el_sayed,
            'stevens': total_stevens,
            'mcmorrow': total_mcmorrow,
            'total': total_votes,
            'el_sayed_pct': total_el_sayed / total_votes,
            'stevens_pct': total_stevens / total_votes,
            'mcmorrow_pct': total_mcmorrow / total_votes,
            'margin': margin
        }
    
    def simulate_confidence_intervals(
        self,
        county_projections_df: pd.DataFrame,
        statewide_shift_estimate: Dict,
        baseline_projection: 'BaselineProjection' = None,
        n_sims: int = None
    ) -> Dict:
        """
        Monte Carlo simulation for confidence intervals.
        
        For each county, sample from Beta-Binomial conditioned on:
        - Expected votes (from projection)
        - Uncertainty proportional to heterogeneity_tau and reporting %
        - Baseline confidence: low-confidence counties get wider uncertainty
        """
        
        n_sims = n_sims or self.config.n_simulations
        tau = statewide_shift_estimate['heterogeneity_tau']
        
        sim_margins = []
        
        for _ in range(n_sims):
            total_el_sayed = 0
            total_stevens = 0
            
            for _, row in county_projections_df.iterrows():
                county_votes = row['el_sayed_projected'] + row['stevens_projected']
                confidence = row.get('baseline_confidence', 1.0)
                
                # Simulate county margin with uncertainty scaled by:
                # 1. Heterogeneity tau
                # 2. County votes (fewer votes = more uncertainty)
                # 3. Baseline confidence (low confidence = higher uncertainty penalty)
                # Confidence acts as inverse multiplier: low confidence increases effective uncertainty
                confidence_penalty = 1.0 / (confidence + 0.01)  # Low conf (0.25) → 4x penalty
                county_uncertainty = tau * confidence_penalty / np.sqrt(county_votes + 1)
                
                simulated_margin = row['adjusted_margin'] + np.random.normal(0, county_uncertainty)
                
                # Convert to votes
                share = (simulated_margin + 100) / 2 / 100
                simulated_el_sayed = int(county_votes * share)
                simulated_stevens = county_votes - simulated_el_sayed
                
                total_el_sayed += simulated_el_sayed
                total_stevens += simulated_stevens
            
            margin = (total_el_sayed - total_stevens) / (total_el_sayed + total_stevens) * 100
            sim_margins.append(margin)
        
        sim_margins = np.array(sim_margins)
        
        return {
            'mean_margin': np.mean(sim_margins),
            'median_margin': np.median(sim_margins),
            'std_margin': np.std(sim_margins),
            'ci_lower': np.percentile(sim_margins, 2.5),
            'ci_upper': np.percentile(sim_margins, 97.5),
            'ci_lower_90': np.percentile(sim_margins, 5.0),
            'ci_upper_90': np.percentile(sim_margins, 95.0),
            'simulations': sim_margins.tolist()
        }


# ============================================================================
# ORCHESTRATION
# ============================================================================

class MichiganSenateModel:
    """Full election night model pipeline"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.baseline_projection = BaselineProjection(
            COUNTY_BASELINES,
            BASELINE_TURNOUT,
            COUNTY_BASELINE_CONFIDENCE
        )
        self.vote_aggregator = LiveVoteAggregator(self.baseline_projection)
        self.shift_estimator = BayesianShiftEstimator(self.config)
        self.county_projector = CountyProjector(
            self.baseline_projection,
            self.vote_aggregator,  # Pass vote aggregator for momentum constraint
            self.config
        )
        self.statewide_projector = StatewideProjector(self.config)
    
    def add_results(self, county: str, el_sayed: int, stevens: int, mcmorrow: int = 0):
        """Add vote totals for a county"""
        self.vote_aggregator.add_county_results(county, el_sayed, stevens, mcmorrow)
    
    def get_projection(self) -> Dict:
        """Generate current projection based on all votes reported so far"""
        
        # Observe what's reported
        observed_margins = self.vote_aggregator.get_observed_margins()
        
        # Estimate statewide shift
        shift_estimate = self.shift_estimator.estimate_statewide_shift(observed_margins)
        
        # Project all counties
        county_projections = self.county_projector.project_all_counties(shift_estimate)
        
        # Aggregate to statewide
        statewide_point = self.statewide_projector.aggregate_statewide(county_projections)
        
        # Simulate confidence intervals (now accounts for county confidence levels)
        statewide_ci = self.statewide_projector.simulate_confidence_intervals(
            county_projections,
            shift_estimate,
            self.baseline_projection
        )
        
        return {
            'timestamp': pd.Timestamp.now().isoformat(),
            'observed_counties': len(observed_margins),
            'statewide_shift': shift_estimate,
            'county_projections': county_projections,
            'statewide_point': statewide_point,
            'statewide_ci': statewide_ci,
            'observed_margins': observed_margins
        }
    
    def export_json(self, projection: Dict, filename: str = None) -> str:
        """Export projection to JSON for web feed"""
        
        export_data = {
            'timestamp': projection['timestamp'],
            'meta': {
                'observed_counties': projection['observed_counties'],
                'statewide_shift_points': round(projection['statewide_shift']['statewide_shift'], 2),
                'shift_ci': [
                    round(projection['statewide_shift']['ci_lower'], 2),
                    round(projection['statewide_shift']['ci_upper'], 2)
                ],
                'heterogeneity_tau': round(projection['statewide_shift']['heterogeneity_tau'], 3)
            },
            'regional_shifts': {},
            'statewide': {
                'point': {
                    'el_sayed': projection['statewide_point']['el_sayed'],
                    'stevens': projection['statewide_point']['stevens'],
                    'mcmorrow': projection['statewide_point']['mcmorrow'],
                    'total': projection['statewide_point']['total'],
                    'el_sayed_pct': round(projection['statewide_point']['el_sayed_pct'], 4),
                    'stevens_pct': round(projection['statewide_point']['stevens_pct'], 4),
                    'mcmorrow_pct': round(projection['statewide_point']['mcmorrow_pct'], 4),
                    'margin': round(projection['statewide_point']['margin'], 2)
                },
                'confidence_intervals': {
                    'margin_95': [
                        round(projection['statewide_ci']['ci_lower'], 2),
                        round(projection['statewide_ci']['ci_upper'], 2)
                    ],
                    'margin_90': [
                        round(projection['statewide_ci']['ci_lower_90'], 2),
                        round(projection['statewide_ci']['ci_upper_90'], 2)
                    ]
                }
            },
            'counties': projection['county_projections'].to_dict('records')
        }
        
        # Add regional shift details if available
        if 'regional_shifts' in projection['statewide_shift']:
            for region, shift_data in projection['statewide_shift']['regional_shifts'].items():
                export_data['regional_shifts'][region] = {
                    'shift': round(shift_data['shift'], 2),
                    'se': round(shift_data['se'], 3),
                    'ci': [round(shift_data['ci_lower'], 2), round(shift_data['ci_upper'], 2)],
                    'n_counties': shift_data['n_counties']
                }
        
        # Remove simulation details for cleaner export
        if 'simulations' in export_data['statewide']['confidence_intervals']:
            del export_data['statewide']['confidence_intervals']['simulations']
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            return filename
        else:
            return json.dumps(export_data, indent=2)


if __name__ == '__main__':
    # Example usage
    model = MichiganSenateModel()
    
    # Print baseline
    baseline = model.baseline_projection.get_statewide_baseline()
    print("BASELINE PROJECTION")
    print(f"El-Sayed: {baseline['el_sayed']:,} ({baseline['el_sayed_pct']:.1%})")
    print(f"Stevens:  {baseline['stevens']:,} ({baseline['stevens_pct']:.1%})")
    print(f"McMorrow: {baseline['mcmorrow']:,} ({baseline['mcmorrow_pct']:.1%})")
    print(f"Margin:   El-Sayed +{baseline['margin']:.1f}")
    print()
