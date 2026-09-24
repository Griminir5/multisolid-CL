"""Regression values are from the original waste_iron_kinetics branch, 9660e5d.

Tests need neither that Git branch nor the original application's legacy API.
"""
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from packed_bed.config import ChemistryConfig, SolidConfig, resolve_case
from packed_bed.definitions import materialize_definitions, select_definitions
from packed_bed.plugins import KineticsContext
from packed_bed.plugins.catalogue import Catalogue
from packed_bed.plugins.check import check_package
from packed_bed.plugins.storage import inspect_package, pack_plugin

PLUGIN = Path(__file__).parents[1] / 'plugins/waste_iron'


def read_catalogue(folder):
    with inspect_package(folder) as package:
        return Catalogue({'waste_iron': package.manifest}, {'waste_iron': package.folder}, {'waste_iron': package.digest})


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    monkeypatch.setenv('MULTISOLID_PLUGIN_APPROVALS', str(tmp_path / 'approvals.json'))
    return read_catalogue(shutil.copytree(PLUGIN, tmp_path / 'plugin'))


def chemistry_and_solids(reactions=('oxidation', 'reduction'), *, alias=False):
    roles = ['O2'] if reactions == ('oxidation',) else ['H2', 'H2O'] if reactions == ('reduction',) else ['O2', 'H2', 'H2O']
    gases = ['gas_' + key if alias else key for key in roles]
    solids = ['iron', 'oxide', 'SiO2', 'Al2O3'] if alias else ['Fe', 'Fe2O3', 'SiO2', 'Al2O3']
    definitions = {'SiO2': 'waste_iron:SiO2', 'Al2O3': 'waste_iron:Al2O3'}
    if alias:
        definitions.update({**dict(zip(gases, ('builtin:' + role for role in roles))), 'iron': 'builtin:Fe', 'oxide': 'builtin:Fe2O3'})
    chemistry = ChemistryConfig(gas_species=gases, species_definitions=definitions,
        reaction_families=['fe_waste'], reaction_ids=['fe_waste/fe_waste_' + name for name in reactions],
        mechanisms={'fe_waste': {'definition': 'waste_iron:fe_waste'}})
    solid = SolidConfig(solid_species=solids, initial_profile={'basis': 'bed', 'zones': [{
        'x_start_m': 0, 'x_end_m': .1, 'e_b': .4, 'e_p': .5, 'd_p': .001,
        'values': dict(zip(solids, (1000, 4500, 1000, 2000)))}]})
    return chemistry, solid


def environment(catalogue, reactions=('oxidation', 'reduction'), *, alias=False):
    chemistry, solids = chemistry_and_solids(reactions, alias=alias)
    return materialize_definitions(select_definitions(chemistry, solids, catalogue), catalogue,
                                   approved=tuple(catalogue.hashes.values()))


def rates(env, temperature, iron, hematite, gas, voidage):
    from daetools.pyDAE import Constant
    from pyUnits import K, m, mol, s
    class Model:
        def T(self, _): return Constant(temperature * K)
        def gasfrac(self, _): return Constant(voidage)
        def c_gas(self, *_): return Constant(gas * voidage * mol / m**3)
        def c_sol(self, index, _): return Constant((iron if index == 0 else hematite) * mol / m**3)
    context = KineticsContext(Model(), 0, {key: i for i, key in enumerate(env.selection.gas_species)},
                              {key: i for i, key in enumerate(env.selection.solid_species)})
    return [hook(context).Node.Quantity.scaleTo(mol / (m**3 * s)).value for hook in env.rate_hooks]


# (T, Fe, Fe2O3, gas concentration per gas volume, reduction extent rate, oxidation extent rate)
BRANCH_RATES = [
    (600, 0, 5000, 1, 0.09598785572671113, 0.0),
    (873.15, 1000, 4500, 10, 8.015800304740827, 29.541617559270758),
    (1100, 5000, 2500, 100, 2.507568887572475, 93.39079563417769),
    (873.15, 9000, 500, .0001, 5.080725890657361e-08, 0.009008299750640696),
    (1100, 9990, 5, 10, 1.1206534450995591e-13, 67.23550381628459),
    (873.15, 10000, 0, 10, 0.0, 67.0350668752896),
]


@pytest.mark.parametrize('voidage', [.2, .7])
@pytest.mark.parametrize('sample', BRANCH_RATES)
def test_rates_match_branch_and_respect_component_bindings(catalogue, sample, voidage):
    env = environment(catalogue, alias=True)
    t, iron, hematite, gas, reduction, oxidation = sample
    np.testing.assert_allclose(rates(env, t, iron, hematite, gas, voidage), [oxidation, reduction], rtol=5e-12, atol=1e-15)
    # Verify Fe/H/O atom balances, including the rebinding of component IDs.
    atoms = {'iron': [1, 0, 0], 'oxide': [2, 0, 3], 'gas_O2': [0, 0, 2],
             'gas_H2': [0, 2, 0], 'gas_H2O': [0, 2, 1]}
    for reaction in env.reaction_network.reactions:
        np.testing.assert_array_equal(sum(n * np.array(atoms[key]) for key, n in reaction.stoichiometry.items()), [0, 0, 0])


@pytest.mark.parametrize('reaction, expected', [('oxidation', 29.541617559270758), ('reduction', 8.015800304740827)])
def test_each_reaction_works_without_unrelated_gases(catalogue, reaction, expected):
    env = environment(catalogue, (reaction,))
    assert rates(env, 873.15, 1000, 4500, 10, .4) == pytest.approx([expected])


@pytest.mark.parametrize('iron, hematite, gas', [(0, 0, 10), (1000, 4500, 0), (-1e-8, -1e-8, 10), (1000, 4500, -1e-8)])
def test_absent_carrier_or_reactant_returns_zero(catalogue, iron, hematite, gas):
    assert rates(environment(catalogue), 900, iron, hematite, gas, .7) == [0, 0]


def test_editable_parameters_are_local_and_fitted_orders_are_fixed(catalogue):
    original = environment(catalogue)
    baseline = rates(original, 873.15, 1000, 4500, 10, .7)
    path = catalogue.paths['waste_iron'] / 'manifest.yaml'
    data = yaml.safe_load(path.read_text())
    data['mechanisms']['fe_waste']['values'] = {'OXIDATION_RATE_CONSTANT_S_INV': 2 * .0073738573592314, 'K0.H2': 0}
    path.write_text(yaml.safe_dump(data))
    modified = environment(read_catalogue(path.parent))
    assert rates(modified, 873.15, 1000, 4500, 10, .7) == pytest.approx([2 * baseline[0], 0])
    assert rates(original, 873.15, 1000, 4500, 10, .7) == baseline
    from packed_bed.parameters import resolve_parameters
    with pytest.raises(ValueError, match='not editable'):
        resolve_parameters(catalogue.manifest('waste_iron').mechanisms['fe_waste'].parameters, {'GAS_REACTION_ORDER.H2': 1})


@pytest.mark.parametrize('key, reference_h, reference_cp', [
    ('SiO2', [-910857.0, -893351.5456409814, -864957.4739855163], [44.8838385, 67.01425113753116, 72.57035138891123]),
    ('Al2O3', [-1675692.0, -1645731.2567922585, -1597640.930949918], [80.2373934, 112.73967630475825, 124.82412995782911]),
])
def test_property_fits_match_branch(catalogue, key, reference_h, reference_cp):
    from daetools.pyDAE import Constant
    from pyUnits import J, K, mol
    correlation = environment(catalogue).properties.get_record(key).enthalpy
    temperatures = [298.15, 600, 1000]
    np.testing.assert_allclose(correlation.value(temperatures), reference_h, rtol=1e-13)
    np.testing.assert_allclose(correlation.cp_value(temperatures), reference_cp, rtol=1e-13)
    symbolic = [correlation.dae_expression(Constant(t * K)).Node.Quantity.scaleTo(J / mol).value for t in temperatures]
    np.testing.assert_allclose(symbolic, reference_h, rtol=1e-13)


def test_archive_loads_in_fresh_process_without_source_folder(catalogue, tmp_path):
    archive = pack_plugin(catalogue.paths['waste_iron'], tmp_path / 'waste_iron.msplugin')
    with pytest.raises(ValueError, match='before execution'):
        check_package(archive)
    shutil.rmtree(catalogue.paths['waste_iron'])
    script = '''
import sys
from packed_bed.plugins.storage import inspect_package
from packed_bed.plugins.check import check_package
with inspect_package(sys.argv[1]) as package:
    check_package(package.folder, approved=(package.digest,))
    assert set(package.manifest.species) == {'SiO2', 'Al2O3'}
    assert len(package.manifest.mechanisms['fe_waste'].reactions) == 2
'''
    result = subprocess.run([sys.executable, '-c', script, str(archive)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('reaction', ['oxidation', 'reduction'])
def test_reactive_case_agrees_between_solver_backends(catalogue, tmp_path, reaction):
    pytest.importorskip('sksundae')
    if shutil.which('g++') is None:
        pytest.skip('Compiled backend requires a C++ compiler.')
    from packed_bed.simulation import run_case
    from packed_bed.reports import load_dataset
    chemistry, solids = chemistry_and_solids((reaction,))
    gases = chemistry.gas_species
    program = {key: {'initial': value, 'steps': []} for key, value in {
        'inlet_flow': .001, 'inlet_temperature': 900, 'outlet_pressure': 100000,
        'inlet_composition': {key: float(key == gases[0]) for key in gases}}.items()}
    run = {
        'references': {key + '_file': key + '.yaml' for key in ('chemistry', 'program', 'solids')},
        'simulation': {'system_name': 'WasteIron', 'time_horizon_s': .1, 'reporting_interval_s': .05,
                       'mass_scheme': 'weno3', 'heat_scheme': 'weno3', 'report_time_derivatives': False},
        'model': {'bed_length_m': .1, 'bed_radius_m': .01, 'axial_cells': 3, 'ambient_temperature_k': 900,
                  'heat_transfer_coefficient_w_per_m2_k': 0},
        # Tighten concentration error as well as relative error for backend comparison.
        'solver': {'name': 'superlu', 'threads': 1, 'relative_tolerance': 1e-9, 'concentration_absolute_tolerance': 1e-9},
        'outputs': {'directory': str(tmp_path / 'daetools'), 'artifacts_directory': str(tmp_path / 'artifacts'), 'requested_plots': [],
                    'requested_reports': ['temperature', 'gas_mole_fraction', 'solid_mole_fraction', 'reaction_rate']},
    }
    case = resolve_case(**{key + '_path': tmp_path / (key + '.yaml') for key in ('run', 'chemistry', 'program', 'solids')},
        run_data=run, chemistry_data=chemistry.model_dump(mode='json'), program_data=program,
        solids_data=solids.model_dump(mode='json'), catalogue=catalogue, approved=tuple(catalogue.hashes.values()))
    datasets = []
    try:
        for backend in ('daetools', 'compiled'):
            solver = case.run.solver.model_copy(update={'backend': backend})
            outputs = case.run.outputs.model_copy(update={'directory': str(tmp_path / backend)})
            resolved = replace(case, run=case.run.model_copy(update={'solver': solver, 'outputs': outputs}))
            result = run_case(resolved)
            datasets.append(load_dataset(result.results_path))
        assert float(datasets[0].reaction_rate.min()) > 0
        assert float(abs(datasets[0].solid_mole_fraction.isel(time=-1) - datasets[0].solid_mole_fraction.isel(time=0)).max()) > 1e-6
        for key in datasets[0].data_vars:
            np.testing.assert_allclose(datasets[0][key], datasets[1][key], rtol=1e-6, atol=1e-8)
    finally:
        for dataset in datasets:
            dataset.close()
