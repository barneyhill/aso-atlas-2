"""
ASO Kinetic Model - Python Implementation
Based on Pedersen et al. (2014) Mol Ther Nucleic Acids

Recreates the RNase H-mediated target degradation model and validates
against cET ASO IC50 data from USPTO patents.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, brentq
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Kinetic Model Parameters (Table 1 from paper)
# ============================================================================
DEFAULT_PARAMS = {
    'Et': 1.0,        # RNase H concentration (nM)
    'KdOT': 0.3,      # Oligo-target dissociation constant (nM)
    'kOpT': 0.2,      # O+T -> OT association rate (nM^-1 min^-1)
    'KdOTE': 70.0,    # OTE dissociation constant (nM)
    'kOTpE': 5.0,     # OT+E -> OTE association rate (nM^-1 min^-1)
    'vprod': 0.2,     # Target production rate (nM/min)
    'kdegrad': 0.04,  # Target degradation rate (min^-1)
    'alpha': 0.1,     # Coupling factor: k_OT->O+T / k_OC->O+C
    'kcleav': 8.0,    # OTE -> OCE cleavage rate (min^-1)
}


# ============================================================================
# Nearest-Neighbor Thermodynamic Parameters
# ============================================================================
# RNA-DNA hybrid parameters from Sugimoto et al. 1995
# dH in kcal/mol, dS in cal/mol/K
NN_PARAMS_DNA_RNA = {
    'AA': (-7.8, -21.9),  'AT': (-5.5, -15.0),  'AG': (-9.1, -25.0),  'AC': (-9.0, -26.1),
    'TA': (-7.8, -23.2),  'TT': (-7.8, -21.9),  'TG': (-5.6, -13.5),  'TC': (-8.6, -22.9),
    'GA': (-8.5, -22.6),  'GT': (-7.0, -19.3),  'GG': (-12.8, -31.9), 'GC': (-11.1, -28.4),
    'CA': (-10.4, -28.4), 'CT': (-9.1, -25.0),  'CG': (-16.3, -47.1), 'CC': (-12.8, -31.9),
    'init': (0.0, -3.1),  # Initiation
}

# NOTE: LNA nearest-neighbor parameters (McTigue et al. 2004) were REMOVED because they
# are for LNA-DNA duplexes, not LNA-RNA. ASOs bind RNA targets, and per PMC7418465:
# "hybridization free energies cannot be accurately predicted for complexes between
# modified gapmer ASOs and RNA" using LNA-DNA parameters.
# We now use only DNA-RNA hybrid parameters from Sugimoto et al. (1995) for Tm prediction.


def calculate_tm(sequence: str, oligo_conc: float = 1e-6, na_conc: float = 0.1) -> float:
    """
    Calculate melting temperature using DNA-RNA nearest-neighbor thermodynamics.

    Uses Sugimoto et al. (1995) parameters for DNA-RNA hybrid duplexes.
    Note: LNA-specific corrections were removed because McTigue 2004 parameters
    are for LNA-DNA duplexes, not LNA-RNA (which is what ASOs bind to).

    Args:
        sequence: DNA sequence (5' to 3')
        oligo_conc: Oligonucleotide concentration (M)
        na_conc: Sodium concentration (M)

    Returns:
        Predicted Tm in Celsius
    """
    sequence = sequence.upper().replace('U', 'T')

    # Calculate dH and dS using nearest-neighbor model
    # dH in kcal/mol, dS in cal/mol/K
    dH = NN_PARAMS_DNA_RNA['init'][0]
    dS = NN_PARAMS_DNA_RNA['init'][1]

    for i in range(len(sequence) - 1):
        dinuc = sequence[i] + sequence[i + 1]
        if dinuc in NN_PARAMS_DNA_RNA:
            dH += NN_PARAMS_DNA_RNA[dinuc][0]
            dS += NN_PARAMS_DNA_RNA[dinuc][1]

    # Salt correction (Owczarzy et al.)
    dS_salt = dS + 0.368 * len(sequence) * np.log(na_conc)

    # Calculate Tm
    R = 1.987  # cal/mol/K
    denominator = dS_salt + R * np.log(oligo_conc / 4)
    if denominator == 0:
        return np.nan
    Tm_K = (dH * 1000) / denominator
    Tm_C = Tm_K - 273.15

    # Sanity check - Tm should be in reasonable range
    if Tm_C < -50 or Tm_C > 150:
        return np.nan

    return Tm_C


def tm_to_kd(tm: float, temp: float = 37.0, dH: float = -120.0) -> float:
    """
    Convert Tm to dissociation constant at specified temperature.

    Uses van't Hoff relationship:
    Kd(T) = Kd(Tm) * exp[(dH/R) * (1/T - 1/Tm)]

    At Tm (measured at 1 µM oligo), Kd ≈ 1 µM = 1000 nM by definition.

    Args:
        tm: Melting temperature (°C)
        temp: Temperature for Kd calculation (°C), default 37°C
        dH: Enthalpy in kcal/mol (typical -100 to -150 for 16-mer ASO)

    Returns:
        Kd in nM
    """
    if np.isnan(tm):
        return np.nan

    R = 1.987e-3  # kcal/mol/K
    T_K = temp + 273.15  # Target temperature in Kelvin
    Tm_K = tm + 273.15   # Melting temperature in Kelvin
    Kd_at_Tm = 1000.0    # nM (1 µM at Tm measurement)

    # Van't Hoff equation
    kd = Kd_at_Tm * np.exp((dH / R) * (1/T_K - 1/Tm_K))

    # Clamp to reasonable range (fM to mM)
    return max(1e-6, min(1e6, kd))


# ============================================================================
# Kinetic Model ODEs
# ============================================================================
def aso_odes(t, y, params):
    """
    ODEs for the ASO kinetic model.

    Species:
        y[0] = T (free target)
        y[1] = OT (oligo-target complex)
        y[2] = OTE (oligo-target-enzyme complex)
        y[3] = E (free enzyme/RNase H)
        y[4] = O (free oligonucleotide)
        y[5] = OCE (cleaved-oligo-enzyme complex)
        y[6] = OC (cleaved-oligo complex)
    """
    T, OT, OTE, E, O, OCE, OC = y

    # Unpack parameters
    Et = params['Et']
    KdOT = params['KdOT']
    kOpT = params['kOpT']
    KdOTE = params['KdOTE']
    kOTpE = params['kOTpE']
    vprod = params['vprod']
    kdegrad = params['kdegrad']
    alpha = params['alpha']
    kcleav = params['kcleav']

    # Derived rate constants
    km1 = kOpT * KdOT      # OT -> O + T dissociation rate
    km2 = kOTpE * KdOTE    # OTE -> OT + E dissociation rate
    k3 = km1 / alpha       # OC -> O + C dissociation rate

    # ODEs (Eqs 2-8 from paper)
    dT = vprod - kOpT*O*T - kdegrad*T + km1*OT
    dOT = kOpT*T*O - km1*OT - kOTpE*OT*E + km2*OTE - kdegrad*OT
    dOTE = kOTpE*OT*E - km2*OTE - kdegrad*OTE - kcleav*OTE
    dE = -kOTpE*OT*E + km2*(OTE + OCE) + kdegrad*OTE
    dO = km1*OT - kOpT*T*O + kdegrad*(OT + OTE) + k3*OC
    dOCE = kcleav*OTE - km2*OCE  # Removed erroneous -k3*OCE term that violated mass conservation
    dOC = km2*OCE - k3*OC

    return [dT, dOT, dOTE, dE, dO, dOCE, dOC]


def simulate_time_course(Ot: float, params: dict = None, t_max: float = 200):
    """Simulate time course of the model."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Initial conditions
    T0 = params['vprod'] / params['kdegrad']
    y0 = [T0, 0, 0, params['Et'], Ot, 0, 0]

    # Solve ODEs
    sol = solve_ivp(
        lambda t, y: aso_odes(t, y, params),
        [0, t_max],
        y0,
        method='LSODA',
        dense_output=True,
        max_step=1.0
    )

    return sol


def calculate_trel_steady_state(Ot: float, params: dict = None) -> float:
    """
    Calculate relative target concentration at steady state.

    Trel = (T + OT + OTE) / T_untreated
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Simulate to steady state
    sol = simulate_time_course(Ot, params, t_max=300)

    # Get steady state values
    T_ss = sol.y[0, -1]
    OT_ss = sol.y[1, -1]
    OTE_ss = sol.y[2, -1]

    # Untreated target level
    T_untreated = params['vprod'] / params['kdegrad']

    # Relative target
    Trel = (T_ss + OT_ss + OTE_ss) / T_untreated

    return max(0, min(1, Trel))


def dose_response_curve(Ot_range: np.ndarray, params: dict = None) -> np.ndarray:
    """Calculate Trel for a range of oligonucleotide concentrations."""
    return np.array([calculate_trel_steady_state(Ot, params) for Ot in Ot_range])


def calculate_ec50(params: dict = None, Ot_range: tuple = (1e-4, 1e3)) -> float:
    """
    Calculate EC50 from dose-response curve.

    EC50 is the oligonucleotide concentration giving half-maximal effect.
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Generate dose-response curve
    Ot_log = np.linspace(np.log10(Ot_range[0]), np.log10(Ot_range[1]), 30)
    Ot_values = 10**Ot_log
    Trel_values = dose_response_curve(Ot_values, params)

    # Find Trel_min (efficacy)
    Trel_min = min(Trel_values)

    # EC50 is where Trel = (1 + Trel_min) / 2
    target_Trel = (1 + Trel_min) / 2

    # Interpolate to find EC50
    try:
        # Find crossing point
        for i in range(len(Trel_values) - 1):
            if Trel_values[i] >= target_Trel >= Trel_values[i+1]:
                # Linear interpolation
                slope = (Ot_values[i+1] - Ot_values[i]) / (Trel_values[i+1] - Trel_values[i])
                ec50 = Ot_values[i] + slope * (target_Trel - Trel_values[i])
                return ec50
    except:
        pass

    return np.nan


# ============================================================================
# IC50 Fitting from Dose-Response Data
# ============================================================================
def hill_equation(x, bottom, top, ic50, hill):
    """4-parameter Hill equation for dose-response fitting."""
    return bottom + (top - bottom) / (1 + (ic50 / x)**hill)


def fit_ic50(doses: np.ndarray, inhibitions: np.ndarray) -> dict:
    """
    Fit IC50 from dose-response data.

    Args:
        doses: Oligonucleotide concentrations (nM)
        inhibitions: Inhibition percentages (0-100, where 100 = full knockdown)

    Returns:
        Dict with IC50, hill coefficient, and fit quality
    """
    # Initial guesses - fit inhibition directly
    p0 = [0, 100, np.median(doses), 1.0]
    bounds = ([-50, 50, 0.01, 0.1], [50, 150, 100000, 10])

    try:
        popt, pcov = curve_fit(
            hill_equation, doses, inhibitions,
            p0=p0, bounds=bounds, maxfev=5000
        )

        # Calculate R²
        fitted = hill_equation(doses, *popt)
        ss_res = np.sum((inhibitions - fitted)**2)
        ss_tot = np.sum((inhibitions - np.mean(inhibitions))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'ic50': popt[2],
            'hill': popt[3],
            'bottom': popt[0],
            'top': popt[1],
            'r2': r2,
            'success': True
        }
    except:
        return {'ic50': np.nan, 'success': False}


# ============================================================================
# HELM Parsing
# ============================================================================
def parse_helm(helm: str) -> dict:
    """
    Parse HELM notation to extract sequence and modifications.

    Returns:
        Dict with sequence, modification positions, etc.
    """
    if pd.isna(helm):
        return None

    # Normalize braces
    helm = helm.replace('{{', '{').replace('}}', '}')

    # Extract nucleotide units
    match = re.search(r'\{(.+?)\}', helm)
    if not match:
        return None

    units = match.group(1).split('.')

    sequence = []
    is_lna = []  # Boolean list: True if position is LNA/cET
    n_cet = 0
    n_moe = 0
    n_lna = 0
    n_dna = 0

    for unit in units:
        # Extract base
        base_match = re.search(r'\(([^)]+)\)', unit)
        if base_match:
            base = base_match.group(1)
            # Normalize 5meC
            if '5me' in base.lower() or '5Me' in base:
                base = 'C'
            sequence.append(base[0].upper())

            # Track if this position is LNA/cET (high affinity)
            if '[cet]' in unit.lower() or '[lna]' in unit.lower():
                is_lna.append(True)
                if '[cet]' in unit.lower():
                    n_cet += 1
                else:
                    n_lna += 1
            elif '[moe]' in unit.lower():
                is_lna.append(False)  # MOE doesn't use LNA NN params
                n_moe += 1
            elif unit.startswith('d('):
                is_lna.append(False)
                n_dna += 1
            else:
                is_lna.append(False)

    return {
        'sequence': ''.join(sequence),
        'is_lna': is_lna,  # Boolean list of LNA positions
        'length': len(sequence),
        'n_cet': n_cet,
        'n_moe': n_moe,
        'n_lna': n_lna,
        'n_dna': n_dna,
        'n_high_affinity': n_cet + n_moe + n_lna,
    }


# ============================================================================
# Main Analysis
# ============================================================================
def load_cet_dose_response_data():
    """Load cET ASO dose-response data from patents."""
    data_path = Path('/Users/barneyh/dphil/paper2/data/oligostack/processed/dose_response_processed.parquet')
    df = pd.read_parquet(data_path)

    # Filter for cET compounds
    cet_mask = df['HELM Annotation'].str.contains('cet', case=False, na=False)
    cet = df[cet_mask].copy()

    print(f"Total cET records: {len(cet)}")
    print(f"Unique cET compounds: {cet['Compound ID'].nunique()}")

    return cet


def calculate_ic50s_for_compounds(df: pd.DataFrame, min_doses: int = 4) -> pd.DataFrame:
    """Calculate IC50 for compounds with sufficient dose points."""
    results = []

    # Group by compound
    grouped = df.groupby('Compound ID')

    for compound_id, group in grouped:
        doses = group['dosage_nm'].values
        inhibitions = group['Inhibition_pct'].values

        # Need at least min_doses points
        if len(doses) < min_doses:
            continue

        # Sort by dose
        sort_idx = np.argsort(doses)
        doses = doses[sort_idx]
        inhibitions = inhibitions[sort_idx]

        # Fit IC50
        fit_result = fit_ic50(doses, inhibitions)

        if fit_result['success'] and fit_result['r2'] > 0.5:
            # Get HELM for this compound
            helm = group['HELM Annotation'].iloc[0]
            parsed = parse_helm(helm)

            if parsed:
                results.append({
                    'Compound ID': compound_id,
                    'ic50_measured': fit_result['ic50'],
                    'hill': fit_result['hill'],
                    'r2': fit_result['r2'],
                    'n_doses': len(doses),
                    'sequence': parsed['sequence'],
                    'length': parsed['length'],
                    'n_cet': parsed['n_cet'],
                    'n_dna': parsed['n_dna'],
                    'helm': helm,
                })

    return pd.DataFrame(results)


def predict_ec50_from_sequence(sequence: str, params: dict = None) -> tuple:
    """
    Predict EC50 from sequence using the kinetic model.

    1. Calculate Tm from sequence using DNA-RNA nearest-neighbor parameters
    2. Convert Tm to Kd
    3. Run kinetic model to get EC50

    Returns:
        (ec50, tm, kd) tuple
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Calculate Tm using DNA-RNA hybrid parameters
    tm = calculate_tm(sequence)

    # Convert to Kd
    kd = tm_to_kd(tm)

    # Update params with this Kd
    params['KdOT'] = kd

    # Calculate EC50
    ec50 = calculate_ec50(params)

    return ec50, tm, kd


def validate_model_single_dose():
    """
    Validate model using single-dose activity data.

    Like Figure 2e-h in the paper, we correlate predicted Tm (binding affinity)
    with activity at a fixed dose to show the optimal affinity phenomenon.
    """
    print("="*60)
    print("ASO Kinetic Model Validation (Single-Dose Approach)")
    print("="*60)

    fig_dir = Path('/Users/barneyh/dphil/paper2/analyses/hypotheses/005_kinetic_model/figures')
    fig_dir.mkdir(exist_ok=True)

    # 1. First, reproduce Figure 2d from paper (EC50 vs KdOT)
    print("\n1. Reproducing Figure 2d: EC50 vs KdOT relationship")

    kd_range = np.logspace(-3, 3, 50)
    ec50_values = []

    for kd in kd_range:
        params = DEFAULT_PARAMS.copy()
        params['KdOT'] = kd
        ec50 = calculate_ec50(params)
        ec50_values.append(ec50)

    ec50_values = np.array(ec50_values)

    fig, ax = plt.subplots(figsize=(8, 6))
    valid_mask = ~np.isnan(ec50_values)
    ax.loglog(kd_range[valid_mask], ec50_values[valid_mask], 'b-', linewidth=2)
    ax.set_xlabel('KdOT (nM)', fontsize=12)
    ax.set_ylabel('EC50 (nM)', fontsize=12)
    ax.set_title('Kinetic Model: EC50 vs Binding Affinity\n(Optimal affinity exists)', fontsize=14)
    ax.grid(True, alpha=0.3)

    opt_idx = np.nanargmin(ec50_values)
    ax.axvline(kd_range[opt_idx], color='red', linestyle='--', alpha=0.7,
               label=f'Optimal KdOT ≈ {kd_range[opt_idx]:.2f} nM')
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / 'ec50_vs_kdot.png', dpi=150)
    plt.close()

    print(f"   Optimal KdOT: {kd_range[opt_idx]:.3f} nM")
    print(f"   Minimum EC50: {ec50_values[opt_idx]:.3f} nM")

    # 2. Load single-dose activity data
    print("\n2. Loading cET ASO single-dose activity data...")
    data_path = Path('/Users/barneyh/dphil/paper2/data/oligostack/processed/in_vitro_inhibition_processed.parquet')
    df = pd.read_parquet(data_path)

    # Filter for cET compounds
    cet = df[df['HELM Annotation'].str.contains('cet', case=False, na=False)].copy()
    print(f"   Total cET records: {len(cet)}")

    # Focus on single concentration for cleaner analysis
    # Use the most common dose
    dose_counts = cet['dosage_nm'].value_counts()
    target_dose = dose_counts.index[0]
    print(f"   Using dose: {target_dose} nM ({dose_counts[target_dose]} compounds)")

    cet_single = cet[cet['dosage_nm'] == target_dose].copy()

    # 3. Calculate predicted Tm for each compound
    print("\n3. Calculating predicted Tm for each compound...")

    results = []
    for _, row in cet_single.iterrows():
        parsed = parse_helm(row['HELM Annotation'])
        if parsed is None:
            continue

        try:
            tm = calculate_tm(parsed['sequence'])
            results.append({
                'Compound ID': row['Compound ID'],
                'inhibition': row['Inhibition_pct'],
                'tm_predicted': tm,
                'sequence': parsed['sequence'],
                'length': parsed['length'],
                'n_cet': parsed['n_cet'],
                'n_dna': parsed['n_dna'],
            })
        except:
            continue

    results_df = pd.DataFrame(results)
    print(f"   Valid compounds: {len(results_df)}")

    # Filter to reasonable Tm range
    results_df = results_df[(results_df['tm_predicted'] > 30) & (results_df['tm_predicted'] < 100)]
    results_df = results_df[(results_df['inhibition'] >= 0) & (results_df['inhibition'] <= 100)]
    print(f"   After filtering: {len(results_df)}")

    # 4. Analyze activity vs Tm (optimal affinity)
    print("\n4. Analyzing activity vs predicted Tm (optimal affinity)...")

    from scipy.stats import spearmanr

    # Bin by Tm and calculate mean activity
    results_df['tm_bin'] = pd.cut(results_df['tm_predicted'], bins=20)
    tm_activity = results_df.groupby('tm_bin').agg({
        'inhibition': ['mean', 'std', 'count'],
        'tm_predicted': 'mean'
    }).dropna()

    tm_activity.columns = ['inhib_mean', 'inhib_std', 'count', 'tm_mean']
    tm_activity = tm_activity[tm_activity['count'] >= 10]  # At least 10 compounds per bin

    print("\n   Activity by Tm bin:")
    print(tm_activity[['tm_mean', 'inhib_mean', 'count']].to_string())

    # Fit quadratic to binned data to show parabola
    if len(tm_activity) >= 5:
        z = np.polyfit(tm_activity['tm_mean'], tm_activity['inhib_mean'], 2)
        p = np.poly1d(z)

        # Find optimal Tm (vertex of parabola)
        optimal_tm = -z[1] / (2 * z[0])

        # Test if it's a significant parabola (negative quadratic term = inverted U)
        is_inverted_u = z[0] < 0

        print(f"\n   Quadratic fit: y = {z[0]:.4f}x² + {z[1]:.4f}x + {z[2]:.4f}")
        print(f"   Optimal Tm: {optimal_tm:.1f}°C")
        print(f"   Inverted U-shape (optimal exists): {is_inverted_u}")

        # Correlation between Tm and activity
        rho, pval = spearmanr(results_df['tm_predicted'], results_df['inhibition'])
        print(f"\n   Spearman correlation (Tm vs activity): ρ = {rho:.3f}, p = {pval:.2e}")

    # 5. Create visualization
    print("\n5. Creating figures...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Model prediction (EC50 vs KdOT)
    ax = axes[0, 0]
    ax.loglog(kd_range[valid_mask], ec50_values[valid_mask], 'b-', linewidth=2)
    ax.axvline(kd_range[opt_idx], color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('KdOT (nM)', fontsize=11)
    ax.set_ylabel('EC50 (nM)', fontsize=11)
    ax.set_title('A. Model: Optimal Binding Affinity Exists', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'Optimal KdOT ≈ {kd_range[opt_idx]:.1f} nM',
            transform=ax.transAxes, fontsize=10, va='top')

    # Panel B: Activity vs Tm scatter
    ax = axes[0, 1]
    scatter = ax.scatter(results_df['tm_predicted'], results_df['inhibition'],
                        alpha=0.3, s=10, c=results_df['length'], cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Length')
    ax.set_xlabel('Predicted Tm (°C)', fontsize=11)
    ax.set_ylabel('Inhibition (%)', fontsize=11)
    ax.set_title('B. Single-Dose Activity vs Predicted Affinity', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Panel C: Binned activity with quadratic fit
    ax = axes[1, 0]
    ax.errorbar(tm_activity['tm_mean'], tm_activity['inhib_mean'],
                yerr=tm_activity['inhib_std']/np.sqrt(tm_activity['count']),
                fmt='ko', capsize=3, markersize=8, label='Binned data')

    if len(tm_activity) >= 5:
        tm_fit = np.linspace(tm_activity['tm_mean'].min(), tm_activity['tm_mean'].max(), 100)
        ax.plot(tm_fit, p(tm_fit), 'r-', linewidth=2, label='Quadratic fit')
        ax.axvline(optimal_tm, color='green', linestyle='--', alpha=0.7,
                   label=f'Optimal Tm ≈ {optimal_tm:.1f}°C')

    ax.set_xlabel('Predicted Tm (°C)', fontsize=11)
    ax.set_ylabel('Mean Inhibition (%)', fontsize=11)
    ax.set_title('C. Optimal Affinity Analysis (Binned)', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel D: Distribution of predicted Tm
    ax = axes[1, 1]
    ax.hist(results_df['tm_predicted'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(results_df['tm_predicted'].median(), color='red', linestyle='--',
               label=f'Median = {results_df["tm_predicted"].median():.1f}°C')
    if len(tm_activity) >= 5:
        ax.axvline(optimal_tm, color='green', linestyle='--',
                   label=f'Optimal = {optimal_tm:.1f}°C')
    ax.set_xlabel('Predicted Tm (°C)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('D. Distribution of Predicted Binding Affinities', fontsize=12)
    ax.legend()

    plt.suptitle(f'ASO Kinetic Model Validation: cET Gapmers at {target_dose} nM\n'
                 f'(n={len(results_df)} compounds)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'model_validation_single_dose.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save results
    results_df.to_csv(fig_dir.parent / 'validation_results_single_dose.csv', index=False)

    print("\n" + "="*60)
    print("Validation Complete!")
    print(f"Figures saved to: {fig_dir}")
    print("="*60)

    return results_df, tm_activity


def validate_model():
    """
    Main validation: compare predicted vs measured IC50.
    """
    print("="*60)
    print("ASO Kinetic Model Validation")
    print("="*60)

    # Create figures directory
    fig_dir = Path('/Users/barneyh/dphil/paper2/analyses/hypotheses/005_kinetic_model/figures')
    fig_dir.mkdir(exist_ok=True)

    # 1. First, reproduce Figure 2d from paper (EC50 vs KdOT)
    print("\n1. Reproducing Figure 2d: EC50 vs KdOT relationship")

    kd_range = np.logspace(-3, 3, 50)
    ec50_values = []

    for kd in kd_range:
        params = DEFAULT_PARAMS.copy()
        params['KdOT'] = kd
        ec50 = calculate_ec50(params)
        ec50_values.append(ec50)

    ec50_values = np.array(ec50_values)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_mask = ~np.isnan(ec50_values)
    ax.loglog(kd_range[valid_mask], ec50_values[valid_mask], 'b-', linewidth=2)
    ax.set_xlabel('KdOT (nM)', fontsize=12)
    ax.set_ylabel('EC50 (nM)', fontsize=12)
    ax.set_title('Kinetic Model: EC50 vs Binding Affinity', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark optimal affinity
    opt_idx = np.nanargmin(ec50_values)
    ax.axvline(kd_range[opt_idx], color='red', linestyle='--', alpha=0.7,
               label=f'Optimal KdOT ≈ {kd_range[opt_idx]:.2f} nM')
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / 'ec50_vs_kdot.png', dpi=150)
    plt.close()

    print(f"   Optimal KdOT: {kd_range[opt_idx]:.3f} nM")
    print(f"   Minimum EC50: {ec50_values[opt_idx]:.3f} nM")

    # 2. Load and process cET data
    print("\n2. Loading cET ASO dose-response data...")
    cet_data = load_cet_dose_response_data()

    # 3. Calculate IC50s
    print("\n3. Fitting IC50 values from dose-response curves...")
    ic50_df = calculate_ic50s_for_compounds(cet_data, min_doses=4)
    print(f"   Successfully fit IC50 for {len(ic50_df)} compounds")

    if len(ic50_df) == 0:
        print("   No compounds with valid IC50 fits!")
        return

    # 4. Predict EC50 for each compound
    print("\n4. Predicting EC50 from sequence using kinetic model...")

    predictions = []
    for _, row in ic50_df.iterrows():
        try:
            ec50_pred, tm, kd = predict_ec50_from_sequence(
                row['sequence'],
                row['n_cet']
            )
            predictions.append({
                'Compound ID': row['Compound ID'],
                'ic50_measured': row['ic50_measured'],
                'ec50_predicted': ec50_pred,
                'tm_predicted': tm,
                'kd_predicted': kd,
                'length': row['length'],
                'n_cet': row['n_cet'],
                'r2_fit': row['r2'],
            })
        except Exception as e:
            continue

    pred_df = pd.DataFrame(predictions)

    # Filter valid predictions
    pred_df = pred_df.dropna(subset=['ic50_measured', 'ec50_predicted'])
    pred_df = pred_df[(pred_df['ic50_measured'] > 0) & (pred_df['ec50_predicted'] > 0)]

    print(f"   Valid predictions: {len(pred_df)}")

    if len(pred_df) < 10:
        print("   Not enough valid predictions for correlation analysis!")
        return pred_df

    # 5. Calculate correlation
    log_measured = np.log10(pred_df['ic50_measured'])
    log_predicted = np.log10(pred_df['ec50_predicted'])

    from scipy.stats import spearmanr, pearsonr

    spearman_r, spearman_p = spearmanr(log_measured, log_predicted)
    pearson_r, pearson_p = pearsonr(log_measured, log_predicted)

    print("\n5. Correlation Analysis:")
    print(f"   Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.2e})")
    print(f"   Pearson r  = {pearson_r:.3f} (p = {pearson_p:.2e})")

    # 6. Plot predicted vs measured
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax = axes[0]
    ax.scatter(pred_df['ic50_measured'], pred_df['ec50_predicted'],
               alpha=0.5, s=30, c=pred_df['n_cet'], cmap='viridis')

    # Add diagonal line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', alpha=0.7, label='y=x')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Measured IC50 (nM)', fontsize=12)
    ax.set_ylabel('Predicted EC50 (nM)', fontsize=12)
    ax.set_title(f'Model Validation\nSpearman ρ = {spearman_r:.3f}, p = {spearman_p:.2e}', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distribution of predicted Tm
    ax = axes[1]
    ax.hist(pred_df['tm_predicted'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(pred_df['tm_predicted'].median(), color='red', linestyle='--',
               label=f'Median Tm = {pred_df["tm_predicted"].median():.1f}°C')
    ax.set_xlabel('Predicted Tm (°C)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Predicted Melting Temperatures', fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / 'model_validation.png', dpi=150)
    plt.close()

    # 7. Analysis by Tm bins (to show optimal affinity effect)
    print("\n6. Activity vs Predicted Tm (optimal affinity analysis):")

    tm_bins = pd.cut(pred_df['tm_predicted'], bins=5)
    tm_activity = pred_df.groupby(tm_bins)['ic50_measured'].agg(['median', 'count'])
    print(tm_activity)

    # Plot activity vs Tm
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(pred_df['tm_predicted'], pred_df['ic50_measured'],
               alpha=0.5, s=30, c=pred_df['n_cet'], cmap='viridis')

    # Fit quadratic to show parabola (optimal affinity)
    z = np.polyfit(pred_df['tm_predicted'], np.log10(pred_df['ic50_measured']), 2)
    p = np.poly1d(z)
    tm_sorted = np.sort(pred_df['tm_predicted'])
    ax.plot(tm_sorted, 10**p(tm_sorted), 'r-', linewidth=2,
            label=f'Quadratic fit')

    # Find optimal Tm
    optimal_tm = -z[1] / (2 * z[0])
    ax.axvline(optimal_tm, color='green', linestyle='--', alpha=0.7,
               label=f'Optimal Tm ≈ {optimal_tm:.1f}°C')

    ax.set_yscale('log')
    ax.set_xlabel('Predicted Tm (°C)', fontsize=12)
    ax.set_ylabel('Measured IC50 (nM)', fontsize=12)
    ax.set_title('IC50 vs Predicted Binding Affinity (Tm)\nShows Optimal Affinity Phenomenon', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / 'ic50_vs_tm.png', dpi=150)
    plt.close()

    print(f"\n   Optimal predicted Tm: {optimal_tm:.1f}°C")

    # Save results
    pred_df.to_csv(fig_dir.parent / 'validation_results.csv', index=False)

    print("\n" + "="*60)
    print("Validation Complete!")
    print(f"Figures saved to: {fig_dir}")
    print("="*60)

    return pred_df


def analyze_16mers_optimal_affinity():
    """
    Focused analysis on 16-mer cET gapmers to show optimal affinity effect.
    """
    print("="*60)
    print("Optimal Affinity Analysis: 16-mer cET Gapmers")
    print("="*60)

    fig_dir = Path('/Users/barneyh/dphil/paper2/analyses/hypotheses/005_kinetic_model/figures')
    fig_dir.mkdir(exist_ok=True)

    # Load results from previous analysis
    results = pd.read_csv(fig_dir.parent / 'validation_results_single_dose.csv')

    # Focus on 16-mers (most common length)
    results_16 = results[results['length'] == 16].copy()
    print(f"Total 16-mer compounds: {len(results_16)}")

    # Filter to reasonable Tm range
    results_16 = results_16[(results_16['tm_predicted'] > 40) & (results_16['tm_predicted'] < 85)]
    print(f"After Tm filtering: {len(results_16)}")

    # Bin by Tm for analysis
    n_bins = 15
    results_16['tm_bin'] = pd.cut(results_16['tm_predicted'], bins=n_bins)

    tm_activity = results_16.groupby('tm_bin', observed=True).agg({
        'inhibition': ['mean', 'std', 'count'],
        'tm_predicted': 'mean'
    }).dropna()

    tm_activity.columns = ['inhib_mean', 'inhib_std', 'count', 'tm_mean']
    tm_activity = tm_activity[tm_activity['count'] >= 50]  # At least 50 compounds per bin

    print("\n   Activity by Tm bin:")
    print(tm_activity[['tm_mean', 'inhib_mean', 'count']].to_string())

    # Fit quadratic to show parabola
    z = np.polyfit(tm_activity['tm_mean'], tm_activity['inhib_mean'], 2)
    p = np.poly1d(z)

    # Find optimal Tm (vertex of parabola)
    optimal_tm = -z[1] / (2 * z[0])
    is_inverted_u = z[0] < 0

    print(f"\n   Quadratic fit: y = {z[0]:.4f}x² + {z[1]:.4f}x + {z[2]:.4f}")
    print(f"   Optimal Tm: {optimal_tm:.1f}°C")
    print(f"   Inverted U-shape (optimal exists): {is_inverted_u}")

    # Statistical test for quadratic term
    from scipy.stats import spearmanr, f_oneway

    # Compare low, optimal, and high Tm groups
    low_tm = results_16[results_16['tm_predicted'] < optimal_tm - 10]['inhibition']
    opt_tm = results_16[(results_16['tm_predicted'] >= optimal_tm - 5) &
                        (results_16['tm_predicted'] <= optimal_tm + 5)]['inhibition']
    high_tm = results_16[results_16['tm_predicted'] > optimal_tm + 10]['inhibition']

    print(f"\n   Activity by Tm group:")
    print(f"   Low Tm (<{optimal_tm-10:.0f}°C):  n={len(low_tm):5d}, mean={low_tm.mean():.1f}%")
    print(f"   Optimal ({optimal_tm-5:.0f}-{optimal_tm+5:.0f}°C): n={len(opt_tm):5d}, mean={opt_tm.mean():.1f}%")
    print(f"   High Tm (>{optimal_tm+10:.0f}°C): n={len(high_tm):5d}, mean={high_tm.mean():.1f}%")

    # ANOVA test
    if len(low_tm) > 10 and len(opt_tm) > 10 and len(high_tm) > 10:
        f_stat, p_val = f_oneway(low_tm, opt_tm, high_tm)
        print(f"\n   ANOVA F-statistic: {f_stat:.2f}, p-value: {p_val:.2e}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Scatter with quadratic fit
    ax = axes[0]
    ax.scatter(results_16['tm_predicted'], results_16['inhibition'],
               alpha=0.2, s=10, c='blue')

    # Binned means with error bars
    ax.errorbar(tm_activity['tm_mean'], tm_activity['inhib_mean'],
                yerr=tm_activity['inhib_std']/np.sqrt(tm_activity['count']),
                fmt='ko', capsize=3, markersize=8, label='Binned means', zorder=5)

    # Quadratic fit
    tm_fit = np.linspace(tm_activity['tm_mean'].min(), tm_activity['tm_mean'].max(), 100)
    ax.plot(tm_fit, p(tm_fit), 'r-', linewidth=2, label='Quadratic fit', zorder=4)

    # Mark optimal
    ax.axvline(optimal_tm, color='green', linestyle='--', alpha=0.7,
               label=f'Optimal Tm = {optimal_tm:.1f}°C', zorder=3)

    ax.set_xlabel('Predicted Tm (°C)', fontsize=12)
    ax.set_ylabel('Inhibition (%)', fontsize=12)
    ax.set_title('16-mer cET Gapmers: Activity vs Predicted Binding Affinity', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 80)
    ax.set_ylim(0, 100)

    # Panel B: Boxplot by Tm quintile
    ax = axes[1]
    results_16['tm_quintile'] = pd.qcut(results_16['tm_predicted'], q=5,
                                        labels=['Q1\n(Low Tm)', 'Q2', 'Q3\n(Medium)',
                                               'Q4', 'Q5\n(High Tm)'])

    boxplot_data = [results_16[results_16['tm_quintile'] == q]['inhibition'].values
                    for q in ['Q1\n(Low Tm)', 'Q2', 'Q3\n(Medium)', 'Q4', 'Q5\n(High Tm)']]

    bp = ax.boxplot(boxplot_data, labels=['Q1\n(Low Tm)', 'Q2', 'Q3\n(Medium)', 'Q4', 'Q5\n(High Tm)'],
                    patch_artist=True)

    colors = ['#ff9999', '#ffcc99', '#99ff99', '#99ccff', '#cc99ff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Binding Affinity Quintile', fontsize=12)
    ax.set_ylabel('Inhibition (%)', fontsize=12)
    ax.set_title('Activity Distribution by Affinity Group\n(Peak activity at intermediate affinity)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean values
    means = [np.mean(d) for d in boxplot_data]
    for i, m in enumerate(means):
        ax.text(i+1, m+2, f'{m:.1f}%', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(fig_dir / 'optimal_affinity_16mers.png', dpi=150)
    plt.savefig(fig_dir / 'optimal_affinity_16mers.svg')
    plt.close()

    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Figure saved to: {fig_dir / 'optimal_affinity_16mers.png'}")
    print("="*60)

    return results_16, tm_activity


def validate_with_patent_ic50s():
    """
    Validate model using IC50 values from US20230167446A1 patent.

    This patent has the most compounds with good dose-response data.
    """
    print("="*60)
    print("Model Validation: Predicted EC50 vs Measured IC50")
    print("Patent: US20230167446A1")
    print("="*60)

    fig_dir = Path('/Users/barneyh/dphil/paper2/analyses/hypotheses/005_kinetic_model/figures')
    fig_dir.mkdir(exist_ok=True)

    # Load dose-response data
    data_path = Path('/Users/barneyh/dphil/paper2/data/oligostack/processed/dose_response_processed.parquet')
    df = pd.read_parquet(data_path)

    # Use US20230167446A1 - has ~400 cET compounds with good dose-response data
    # Filter to single condition: Gymnosis in A431 cells for clean comparison
    # Exclude compounds with 2'-OMe (m()) - we don't have params for these
    patent = 'US20230167446A1'
    patent_df = df[(df['USPTO ID'] == patent) &
                   df['HELM Annotation'].str.contains('cet', case=False, na=False) &
                   ~df['HELM Annotation'].str.contains(r'm\(', case=False, na=False, regex=True) &
                   (df['transfection_method'] == 'Gymnosis') &
                   (df['cell_line'] == 'A431')].copy()

    print(f"\nTotal records in {patent} (Gymnosis/A431, no 2'-OMe): {len(patent_df)}")
    print(f"Unique compounds: {patent_df['Compound ID'].nunique()}")

    # Group by compound and fit IC50
    print("\nFitting IC50 values...")

    ic50_results = []
    grouped = patent_df.groupby('Compound ID')

    for compound_id, group in grouped:
        # Aggregate replicates at each dose
        dose_response = group.groupby('dosage_nm').agg({
            'Inhibition_pct': 'mean'
        }).reset_index()

        doses = dose_response['dosage_nm'].values
        inhibitions = dose_response['Inhibition_pct'].values

        # Need at least 4 doses with 100x range
        if len(doses) < 4:
            continue
        dose_range = doses.max() / doses.min()
        if dose_range < 100:
            continue

        # Sort by dose
        sort_idx = np.argsort(doses)
        doses = doses[sort_idx]
        inhibitions = inhibitions[sort_idx]

        # Check for reasonable dose-response (higher dose should generally = higher inhibition)
        low_dose_inhib = np.mean(inhibitions[:3]) if len(inhibitions) >= 3 else inhibitions[0]
        high_dose_inhib = np.mean(inhibitions[-3:]) if len(inhibitions) >= 3 else inhibitions[-1]
        if high_dose_inhib < low_dose_inhib:
            continue  # Response goes wrong direction

        # Fit IC50
        fit_result = fit_ic50(doses, inhibitions)

        if fit_result['success'] and fit_result['r2'] > 0.7:  # Quality threshold
            helm = group['HELM Annotation'].iloc[0]
            parsed = parse_helm(helm)

            if parsed and parsed['length'] > 0:
                ic50_results.append({
                    'Compound ID': compound_id,
                    'ic50_measured': fit_result['ic50'],
                    'hill': fit_result['hill'],
                    'r2': fit_result['r2'],
                    'n_doses': len(doses),
                    'dose_range': dose_range,
                    'sequence': parsed['sequence'],
                    'length': parsed['length'],
                    'n_cet': parsed['n_cet'],
                    'n_dna': parsed['n_dna'],
                    'helm': helm,
                })

    ic50_df = pd.DataFrame(ic50_results)
    print(f"Compounds with valid IC50 fit (R² > 0.7): {len(ic50_df)}")

    if len(ic50_df) < 10:
        print("Not enough compounds for analysis!")
        return None

    # Calculate predicted EC50 for each compound
    print("\nCalculating predicted EC50 from kinetic model...")

    predictions = []
    for _, row in ic50_df.iterrows():
        try:
            # Re-parse HELM to get is_lna positions
            parsed = parse_helm(row['helm'])
            if not parsed:
                continue

            # Calculate Tm using DNA-RNA nearest-neighbor parameters
            tm = calculate_tm(parsed['sequence'])

            # Convert to Kd
            kd = tm_to_kd(tm)

            # Calculate EC50 from model
            params = DEFAULT_PARAMS.copy()
            params['KdOT'] = kd
            ec50_pred = calculate_ec50(params)

            predictions.append({
                'Compound ID': row['Compound ID'],
                'ic50_measured': row['ic50_measured'],
                'ec50_predicted': ec50_pred,
                'tm_predicted': tm,
                'kd_predicted': kd,
                'length': row['length'],
                'n_cet': row['n_cet'],
                'r2_fit': row['r2'],
                'sequence': row['sequence'],
            })
        except Exception as e:
            continue

    pred_df = pd.DataFrame(predictions)

    # Filter valid predictions
    pred_df = pred_df.dropna(subset=['ic50_measured', 'ec50_predicted'])
    pred_df = pred_df[(pred_df['ic50_measured'] > 0) & (pred_df['ec50_predicted'] > 0)]
    pred_df = pred_df[(pred_df['ic50_measured'] < 100000) & (pred_df['ec50_predicted'] < 100000)]

    print(f"Valid predictions: {len(pred_df)}")

    # Correlation analysis
    from scipy.stats import spearmanr, pearsonr

    log_measured = np.log10(pred_df['ic50_measured'])
    log_predicted = np.log10(pred_df['ec50_predicted'])

    spearman_r, spearman_p = spearmanr(log_measured, log_predicted)
    pearson_r, pearson_p = pearsonr(log_measured, log_predicted)

    # Also correlate Tm with IC50 (should show optimal affinity)
    tm_ic50_r, tm_ic50_p = spearmanr(pred_df['tm_predicted'], np.log10(pred_df['ic50_measured']))

    print(f"\n{'='*40}")
    print("Correlation Results:")
    print(f"{'='*40}")
    print(f"Predicted EC50 vs Measured IC50:")
    print(f"  Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.2e})")
    print(f"  Pearson r  = {pearson_r:.3f} (p = {pearson_p:.2e})")
    print(f"\nPredicted Tm vs Measured IC50:")
    print(f"  Spearman ρ = {tm_ic50_r:.3f} (p = {tm_ic50_p:.2e})")

    # Create figure with multiple panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Predicted EC50 vs Measured IC50
    ax = axes[0, 0]
    scatter = ax.scatter(pred_df['ic50_measured'], pred_df['ec50_predicted'],
                        alpha=0.6, s=40, c=pred_df['tm_predicted'], cmap='coolwarm',
                        edgecolors='black', linewidth=0.3)
    plt.colorbar(scatter, ax=ax, label='Predicted Tm (°C)')

    # Add diagonal line
    lims = [1, 10000]
    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=2, label='y = x')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Measured IC50 (nM)', fontsize=12)
    ax.set_ylabel('Predicted EC50 (nM)', fontsize=12)
    ax.set_title(f'A. Model Prediction vs Measurement\nSpearman ρ = {spearman_r:.3f} (p = {spearman_p:.2e})', fontsize=11)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel B: IC50 vs Predicted Tm (optimal affinity test)
    ax = axes[0, 1]
    ax.scatter(pred_df['tm_predicted'], pred_df['ic50_measured'],
               alpha=0.6, s=40, c=pred_df['n_cet'], cmap='viridis',
               edgecolors='black', linewidth=0.3)

    # Fit quadratic to show optimal affinity
    valid = pred_df['tm_predicted'].notna() & (pred_df['ic50_measured'] > 0)
    tm_valid = pred_df.loc[valid, 'tm_predicted'].values
    ic50_valid = pred_df.loc[valid, 'ic50_measured'].values

    if len(tm_valid) > 10:
        z = np.polyfit(tm_valid, np.log10(ic50_valid), 2)
        p = np.poly1d(z)
        tm_fit = np.linspace(tm_valid.min(), tm_valid.max(), 100)
        ax.plot(tm_fit, 10**p(tm_fit), 'r-', linewidth=2, label='Quadratic fit')

        optimal_tm = -z[1] / (2 * z[0]) if z[0] != 0 else np.nan
        if 40 < optimal_tm < 90:
            ax.axvline(optimal_tm, color='green', linestyle='--', alpha=0.7,
                       label=f'Optimal Tm ≈ {optimal_tm:.1f}°C')

    ax.set_yscale('log')
    ax.set_xlabel('Predicted Tm (°C)', fontsize=12)
    ax.set_ylabel('Measured IC50 (nM)', fontsize=12)
    ax.set_title(f'B. IC50 vs Binding Affinity\nρ = {tm_ic50_r:.3f} (p = {tm_ic50_p:.2e})', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Distribution of IC50s
    ax = axes[1, 0]
    ax.hist(np.log10(pred_df['ic50_measured']), bins=30, edgecolor='black',
            alpha=0.7, color='steelblue', label='Measured IC50')
    ax.hist(np.log10(pred_df['ec50_predicted']), bins=30, edgecolor='black',
            alpha=0.5, color='coral', label='Predicted EC50')
    ax.axvline(np.log10(pred_df['ic50_measured'].median()), color='blue',
               linestyle='--', linewidth=2, label=f'Median measured: {pred_df["ic50_measured"].median():.0f} nM')
    ax.axvline(np.log10(pred_df['ec50_predicted'].median()), color='red',
               linestyle='--', linewidth=2, label=f'Median predicted: {pred_df["ec50_predicted"].median():.0f} nM')
    ax.set_xlabel('log₁₀(IC50 or EC50) (nM)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('C. Distribution of IC50/EC50 Values', fontsize=11)
    ax.legend(fontsize=9)

    # Panel D: Residuals by Tm
    ax = axes[1, 1]
    residuals = np.log10(pred_df['ec50_predicted']) - np.log10(pred_df['ic50_measured'])
    ax.scatter(pred_df['tm_predicted'], residuals, alpha=0.6, s=30,
               c=pred_df['length'], cmap='plasma', edgecolors='black', linewidth=0.2)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.axhline(residuals.mean(), color='green', linestyle=':', linewidth=2,
               label=f'Mean residual: {residuals.mean():.2f}')
    ax.set_xlabel('Predicted Tm (°C)', fontsize=12)
    ax.set_ylabel('log₁₀(Predicted) - log₁₀(Measured)', fontsize=12)
    ax.set_title(f'D. Prediction Residuals\nMean = {residuals.mean():.2f}, Std = {residuals.std():.2f}', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Kinetic Model Validation: {patent}\n({len(pred_df)} cET compounds with IC50 fit R² > 0.7)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'patent_ic50_validation.png', dpi=150, bbox_inches='tight')
    plt.savefig(fig_dir / 'patent_ic50_validation.svg', bbox_inches='tight')
    plt.close()

    # Summary statistics
    print(f"\n{'='*40}")
    print("Summary Statistics:")
    print(f"{'='*40}")
    print(f"Compounds analyzed: {len(pred_df)}")
    print(f"Measured IC50: median = {pred_df['ic50_measured'].median():.1f} nM, "
          f"range = {pred_df['ic50_measured'].min():.1f} - {pred_df['ic50_measured'].max():.1f} nM")
    print(f"Predicted EC50: median = {pred_df['ec50_predicted'].median():.1f} nM, "
          f"range = {pred_df['ec50_predicted'].min():.1f} - {pred_df['ec50_predicted'].max():.1f} nM")
    print(f"Predicted Tm: median = {pred_df['tm_predicted'].median():.1f}°C, "
          f"range = {pred_df['tm_predicted'].min():.1f} - {pred_df['tm_predicted'].max():.1f}°C")

    # Save results
    pred_df.to_csv(fig_dir.parent / 'patent_ic50_validation_results.csv', index=False)

    print(f"\nResults saved to: {fig_dir.parent / 'patent_ic50_validation_results.csv'}")
    print(f"Figure saved to: {fig_dir / 'patent_ic50_validation.png'}")

    return pred_df


if __name__ == '__main__':
    # Validate using IC50 values from patent
    pred_df = validate_with_patent_ic50s()
