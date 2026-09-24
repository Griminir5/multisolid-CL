"""N2 + 3 H2 -> 2 NH3; a teaching rate law, not a Haber-Bosch model."""
from packed_bed.plugins import ParameterSpec, ReactionDefinition, ReactionFamily

PARAMETERS = {
    'k': ParameterSpec(0.001, 'm^3 bed/(mol*s)', 'Rate coefficient using concentrations per bed volume', minimum=0),
}


def rate(context):
    from daetools.pyDAE import Constant, Max
    from pyUnits import m, mol, s
    model, cell = context.model, context.idx_cell
    zero = Constant(0 * mol / m**3)
    nitrogen = Max(model.c_gas(context.gas_index('N2'), cell), zero)
    hydrogen = Max(model.c_gas(context.gas_index('H2'), cell), zero)
    return Constant(context.parameters['k'] * m**3 / (mol * s)) * nitrogen * hydrogen


def create(parameters):
    reaction = ReactionDefinition(
        id='synthesis', name='N2 + 3 H2 → 2 NH3', phase='gas_gas',
        stoichiometry={'N2': -1, 'H2': -3, 'NH3': 2}, required_species=('N2', 'H2', 'NH3'),
        source_reference='Illustrative plugin example',
        notes='Irreversible, first order in each reactant. No catalyst, equilibrium or pressure fit.',
    )
    return ReactionFamily('Ammonia synthesis', (reaction,), ('N2', 'H2', 'NH3'), (), {'synthesis': rate})
