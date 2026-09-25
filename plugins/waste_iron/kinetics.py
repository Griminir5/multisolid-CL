"""Waste iron rates ported from waste_iron_kinetics at 9660e5d.

Gas concentrations are per gas volume; solid inventories and returned rates are
per total bed volume. See README.md for the equations and source history.
"""
from daetools.pyDAE import Abs, Constant, Exp, Log, Max, Sqrt
from pyUnits import K, m, mol, s
from packed_bed.plugins import ParameterSpec, ReactionDefinition, ReactionFamily

PARAMETERS = {
    'K0.H2': ParameterSpec(0.1382, 's^-1 (mol/m^3 gas)^-0.7554', 'H2 reduction prefactor', minimum=0),
    'ACTIVATION_ENERGY_J_PER_MOL.H2': ParameterSpec(37318.0, 'J/mol', 'H2 reduction activation energy', minimum=0),
    'K_MT.H2': ParameterSpec(0.0180, 's^-1', 'Reduction mass-transfer rate limit', minimum=1e-30),
    'OXIDATION_RATE_CONSTANT_S_INV': ParameterSpec(0.0073738573592314, 's^-1', 'Oxidation rate constant', minimum=0),
    'OXIDATION_O2_HALF_SATURATION_MOLM3': ParameterSpec(1.0, 'mol/m^3 gas', 'O2 half-saturation concentration', minimum=1e-30),
    'OXIDATION_RANDOM_PORE_PSI': ParameterSpec(8.0, 'dimensionless', 'Oxidation random-pore parameter', minimum=0),
}

# These belong to the fitted rate shape, not its editable parameter contract.
GAS_CONSTANT = 8.31446261815324
H2_ORDER = 0.7554
H2_OFFSET = 0.0001
REMAINING_EPS = 1.0e-10
RATIONAL = (1.452, 11.849, 1.038, 0.170)
PADE_NUMER = (0.2476, 67.3172, 43.4567)
PADE_DENOM = (1.0, 92.6243)


def _positive(value):
    return Constant(0.5) * (value + Abs(value))


def _solid_concentrations(context):
    return tuple(_positive(context.model.c_sol(context.solid_index(role), context.idx_cell)
                           / Constant(1 * mol / m**3)) for role in ('Fe', 'Fe2O3'))


def _gas_concentration(context, role):
    return _positive(context.model.c_gas(context.gas_index(role), context.idx_cell)
                     / context.model.gasfrac(context.idx_cell) / Constant(1 * mol / m**3))


def oxidation(context):
    parameters = context.parameters
    iron, hematite = _solid_concentrations(context)
    sites = hematite + 0.5 * iron
    conversion = hematite / Max(sites, Constant(1e-30))
    remaining = _positive(1 - conversion)
    pore_term = 1 - parameters['OXIDATION_RANDOM_PORE_PSI'] * Log(remaining + REMAINING_EPS)
    solid_term = remaining * Sqrt(Max(pore_term, Constant(0)))
    oxygen = _gas_concentration(context, 'O2')
    gas_term = oxygen / (oxygen + parameters['OXIDATION_O2_HALF_SATURATION_MOLM3'])
    return Constant(1 * mol / (m**3 * s)) * 2 * sites * parameters['OXIDATION_RATE_CONSTANT_S_INV'] * solid_term * gas_term


def reduction(context):
    parameters = context.parameters
    iron, hematite = _solid_concentrations(context)
    sites = iron + 2 * hematite
    conversion = iron / Max(sites, Constant(1e-30))
    remaining = 1 - conversion
    a, b, c, d = RATIONAL
    rational = (a * remaining / (1 + b * Abs(remaining)) + c * remaining / (1 + d * Abs(remaining))) / (a / (1 + b) + c / (1 + d))
    solid_term = rational * remaining**4
    avrami = (PADE_NUMER[0] + PADE_NUMER[1] * conversion + PADE_NUMER[2] * conversion**2) / (PADE_DENOM[0] + PADE_DENOM[1] * conversion)
    hydrogen = _gas_concentration(context, 'H2')
    temperature = context.model.T(context.idx_cell) / Constant(1 * K)
    k_rxn = parameters['K0']['H2'] * Exp(-parameters['ACTIVATION_ENERGY_J_PER_MOL']['H2'] / (GAS_CONSTANT * temperature))
    k_rxn *= (hydrogen + H2_OFFSET)**H2_ORDER - H2_OFFSET**H2_ORDER
    k_eff = k_rxn / (1 + k_rxn / parameters['K_MT']['H2'])
    return Constant(1 * mol / (m**3 * s)) * 0.5 * sites * k_eff * solid_term * avrami


def create(parameters):
    source = 'Waste iron branch 9660e5d; Aya data; Solid-State kinetic modelling (iron waste based catalyst).pdf'
    reactions = (
        ReactionDefinition(
            id='fe_waste_oxidation', name='Waste iron oxidation by O2', phase='gas_solid',
            stoichiometry={'Fe': -1.0, 'O2': -0.75, 'Fe2O3': 0.5},
            required_species=('Fe', 'O2', 'Fe2O3'), source_reference=source,
            notes='One-step Fe to Fe2O3 random-pore oxidation with saturating O2 availability.',
        ),
        ReactionDefinition(
            id='fe_waste_reduction', name='Waste hematite reduction by H2', phase='gas_solid',
            stoichiometry={'Fe2O3': -1.0, 'H2': -3.0, 'Fe': 2.0, 'H2O': 3.0},
            required_species=('Fe', 'H2', 'Fe2O3', 'H2O'), source_reference=source,
            notes='One-step Fe2O3 to Fe reduction with fitted rational/Padé solid terms and a mass-transfer rate limit.',
        ),
    )
    return ReactionFamily('Waste iron redox', reactions, ('O2', 'H2', 'H2O'), ('Fe', 'Fe2O3'),
                          {'fe_waste_oxidation': oxidation, 'fe_waste_reduction': reduction})
