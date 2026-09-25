"""Display associations for built-in parameters; values stay in their declarations.

These describe which declared parameters each existing rate reads. They do not
change the editing contract or the resolved scientific inputs. Third-party
implementations without per-reaction metadata show their family parameter list.
"""


def reaction_parameter_groups(definition):
    parameters = definition.parameters

    def keys(*prefixes):
        return tuple(key for key in parameters if any(key == prefix or key.startswith(prefix + '.') for prefix in prefixes))

    def coefficients(reactions, *groups):
        return {reaction: keys(*(group + '.' + rate for group in groups)) for reaction, rate in reactions.items()}

    implementation = definition.implementation
    if implementation == 'builtin:nickel_medrano':
        return coefficients({
            'ni_reduction_h2_medrano': 'H2', 'ni_reduction_co_medrano': 'CO', 'ni_oxidation_o2_medrano': 'O2',
        }, 'R0_M', 'K0_M_PER_S', 'ACTIVATION_ENERGY_J_PER_MOL')
    if implementation == 'builtin:copper_sio2_san_pio':
        return {
            **coefficients({
                'cuo_h2_reduction_sio2_san_pio': 'cuo', 'cu2o_h2_reduction_sio2_san_pio': 'cu2o',
            }, 'REDUCTION_COEFFICIENTS', 'REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL'),
            **coefficients({
                'cu_sio2_oxidation_1_san_pio': 'cu_to_cuo',
            }, 'OXIDATION_COEFFICIENTS', 'OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL'),
        }
    if implementation == 'builtin:copper_al2o3_san_pio':
        return {
            **coefficients({
                'cuo_h2_reduction_al2o3_san_pio': 'cuo', 'cu2o_h2_reduction_al2o3_san_pio': 'cu2o',
                'cu_al2o3_spinel_reduction_1_san_pio': 'spinel_to_cu',
                'cu_al2o3_spinel_reduction_2_san_pio': 'spinel_to_cualo2',
                'cu_al2o3_spinel_reduction_3_san_pio': 'cualo2_to_cu',
            }, 'REDUCTION_COEFFICIENTS', 'REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL'),
            **coefficients({
                'cu_al2o3_oxidation_1_san_pio': 'cu_to_cuo',
                'cu_al2o3_oxidation_2_san_pio': 'cuo_to_spinel',
                'cu_al2o3_oxidation_3_san_pio': 'cualo2_to_spinel',
            }, 'OXIDATION_COEFFICIENTS', 'OXIDATION_ACTIVATION_ENERGIES_J_PER_MOL'),
        }
    if implementation == 'builtin:reforming_numaguchi':
        return coefficients({
            'smr_reaction_numaguchi': 'smr', 'wgs_reaction_numaguchi': 'wgs',
        }, 'NUMAGUCHI_RATE_COEFFICIENTS', 'NUMAGUCHI_ACTIVATION_ENERGIES_J_PER_MOL')
    if implementation == 'builtin:reforming_xu_froment':
        shared = keys('XU_FROMENT_ADSORPTION_COEFFICIENTS', 'XU_FROMENT_ADSORPTION_ENERGIES_J_PER_MOL')
        specific = coefficients({
            'smr_reaction_xu_froment': 'smr', 'wgs_reaction_xu_froment': 'wgs', 'overall_reforming_xu_froment': 'overall',
        }, 'XU_FROMENT_RATE_COEFFICIENTS', 'XU_FROMENT_ACTIVATION_ENERGIES_J_PER_MOL')
        return {reaction: own + shared for reaction, own in specific.items()}
    if implementation == 'builtin:iron_he':
        return {
            **{f'{solid.lower()}_h2_reduction_he_2023': keys('AVERAGE_GRAIN_RADIUS_M',
                f'H2_REDUCTION_PREEXPONENTIALS.{solid}', f'H2_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL.{solid}',
                f'H2_DIFFUSIVITY_PREEXPONENTIALS.{solid}', f'H2_DIFFUSIVITY_ACTIVATION_ENERGIES_J_PER_MOL.{solid}')
               for solid in ('Fe2O3', 'Fe3O4', 'FeO')},
            **{f'{solid.lower()}_co_reduction_he_2023': keys(f'CO_REDUCTION_ACTIVATION_ENERGIES_J_PER_MOL.{solid}')
               for solid in ('Fe2O3', 'Fe3O4', 'FeO')},
            'fe2o3_ch4_reduction_he_2023': keys('CH4_REDUCTION_PREEXPONENTIAL', 'CH4_REDUCTION_ACTIVATION_ENERGY_J_PER_MOL'),
            'fe_o2_oxidation_he_2023': (),
        }
    return None
