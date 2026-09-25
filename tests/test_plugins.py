from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import yaml

from packed_bed.config.models import ChemistryConfig, SolidConfig
from packed_bed.definitions import select_definitions, materialize_definitions
from packed_bed.parameters import freeze, plain, resolve_parameters
from packed_bed.plugins.catalogue import Catalogue, builtin_manifest
from packed_bed.plugins.check import check_package, generate_metadata
from packed_bed.plugins.schema import Manifest
from packed_bed.plugins.storage import (inspect_package, copy_package, catalogue_from_project,
                                        package_hash, pack_plugin, approve_hash, code_hash)

EXAMPLES = Path(__file__).parents[1] / 'packed_bed/examples/plugins'


@pytest.fixture(autouse=True)
def isolated_approval(monkeypatch, tmp_path):
    monkeypatch.setenv('MULTISOLID_PLUGIN_APPROVALS', str(tmp_path / 'approvals.json'))


def package(tmp_path, *, ident='custom', species=None, mechanisms=None):
    folder = tmp_path / ident
    folder.mkdir(exist_ok=True)
    data = {'id': ident, 'name': 'Custom properties', 'species': species or {}, 'mechanisms': mechanisms or {}}
    (folder / 'manifest.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    return folder


def catalogue(folder, tmp_path):
    entry = copy_package(folder, tmp_path / 'project')
    return catalogue_from_project(tmp_path / 'project', [{**entry, 'enabled': True}])


def inputs(gases, solids, families=(), reactions=(), **kwargs):
    chemistry = ChemistryConfig(gas_species=gases, reaction_families=families, reaction_ids=reactions, **kwargs)
    solid = SolidConfig.model_validate({'solid_species': solids, 'initial_profile': {'basis': 'bed', 'zones': [
        {'x_start_m': 0.0, 'x_end_m': 1.0, 'e_b': .4, 'e_p': .5, 'd_p': .001, 'values': {key: 1.0 for key in solids}}]}})
    return chemistry, solid


def test_two_waters_keep_separate_properties_and_require_unambiguous_binding(tmp_path):
    water = builtin_manifest().species['H2O'].model_dump(mode='json')
    water['enthalpy']['parameters']['h_form_ref'] += 5000
    cat = catalogue(package(tmp_path, species={'water': water}), tmp_path)
    chemistry, solids = inputs(['water_a', 'water_b', 'H2'], ['Ni', 'NiO'], ['nickel_medrano'], ['ni_reduction_h2_medrano'],
        species_definitions={'water_a': 'builtin:H2O', 'water_b': 'custom:water'})
    with pytest.raises(ValueError, match='bind H2O explicitly'):
        select_definitions(chemistry, solids, cat)
    chemistry = chemistry.model_copy(update={'mechanisms': {
        'nickel_medrano': __import__('packed_bed.config.models', fromlist=['MechanismSelection']).MechanismSelection(
            definition='builtin:nickel_medrano', bindings={'H2O': 'water_b'})}})
    environment = materialize_definitions(select_definitions(chemistry, solids, cat), cat)
    assert environment.reaction_network.reactions[0].stoichiometry['water_b'] == 1
    assert 'water_a' not in environment.reaction_network.reactions[0].stoichiometry
    assert environment.properties.enthalpy_value('water_b', 500) - environment.properties.enthalpy_value('water_a', 500) == pytest.approx(5000)
    assert 'Custom properties' in environment.selection.labels['water_b']
    with pytest.raises(TypeError):
        environment.selection.species['water_b']['definition']['mw'] = 1
    with pytest.raises(ValueError, match='properties'):
        replace(environment, properties=replace(environment.properties, records={'H2': environment.properties.get_record('H2')}))
    with pytest.raises(ValueError, match='bindings differ'):
        environment.matches(chemistry.model_copy(update={'mechanisms': {}}), solids)
    records = dict(environment.properties.records)
    records['water_b'] = replace(records['water_b'], enthalpy=records['water_a'].enthalpy)
    with pytest.raises(ValueError, match='enthalpy'):
        replace(environment, properties=replace(environment.properties, records=records))


def test_parameter_instances_do_not_mutate_globals_and_selected_ni_mw_reaches_rates(tmp_path):
    from packed_bed.reactions import KineticsContext
    from packed_bed.kinetics.reforming_xu_froment import XU_FROMENT_RATE_COEFFICIENTS
    original = dict(XU_FROMENT_RATE_COEFFICIENTS)
    definition = builtin_manifest().mechanisms['reforming_xu_froment'].model_dump(mode='json')
    definition['values'] = {'XU_FROMENT_RATE_COEFFICIENTS.smr': original['smr'] * 2}
    nickel = builtin_manifest().species['Ni'].model_dump(mode='json')
    nickel['mw'] *= 3
    cat = catalogue(package(tmp_path, species={'nickel': nickel}, mechanisms={'reforming': definition}), tmp_path)
    gas = ['CH4', 'H2O', 'CO', 'CO2', 'H2']
    chemistry, solids = inputs(gas, ['Ni'], ['reforming_xu_froment', 'variant'],
        ['smr_xu_froment', 'variant/smr_xu_froment'], mechanisms={'variant': {'definition': 'custom:reforming'}})
    # Use the actual legacy reaction ID from its single implementation contract.
    local = definition['reactions'][0]['id']
    chemistry = chemistry.model_copy(update={'reaction_ids': (local, 'variant/' + local)})
    baseline = materialize_definitions(select_definitions(chemistry, solids, cat), cat)
    changed = chemistry.model_copy(update={'species_definitions': {'Ni': 'custom:nickel'}})
    heavy = materialize_definitions(select_definitions(changed, solids, cat), cat)
    from daetools.pyDAE import Constant
    from pyUnits import K, Pa, mol, m
    class Model:
        def T(self, _): return Constant(900 * K)
        def P(self, _): return Constant(200000 * Pa)
        def c_sol(self, *_): return Constant(1000 * mol / m**3)
        def y_gas(self, index, _): return Constant([.3, .4, .01, .01, .28][index])
    context = KineticsContext(Model(), 0, {key: i for i, key in enumerate(gas)}, {'Ni': 0})
    first, second = [hook(context).Node.Quantity.value for hook in baseline.rate_hooks]
    assert second == pytest.approx(2 * first)
    assert heavy.rate_hooks[0](context).Node.Quantity.value == pytest.approx(3 * first)
    assert XU_FROMENT_RATE_COEFFICIENTS == original
    with pytest.raises(ValueError, match='reaction order'):
        replace(baseline, rate_hooks=baseline.rate_hooks[::-1])
    with pytest.raises(ValueError, match='supplied environment'):
        baseline.matches(changed, solids)


@pytest.mark.parametrize('temperature,pressure,oxygen,copper', [
    (873.15, 1e5, 0.21, 1000.0),
    (1073.15, 2e5, 0.05, 3000.0),
    (1273.15, 1e5, 0.0, 1000.0),
    (873.15, 1e5, -1e-8, 1000.0),
    (1073.15, 1e5, 0.21, 0.0),
])
def test_copper_oxidation_rate_law_matches_between_supports(temperature, pressure, oxygen, copper):
    from daetools.pyDAE import Constant
    from pyUnits import K, Pa, mol, m, s
    from packed_bed.kinetics import copper_al2o3, copper_sio2
    from packed_bed.reactions import KineticsContext

    class Model:
        def T(self, _): return Constant(temperature * K)
        def P(self, _): return Constant(pressure * Pa)
        def y_gas(self, index, _): return Constant([oxygen][index])
        def c_sol(self, index, _): return Constant([copper][index] * mol / m**3)

    context = KineticsContext(Model(), 0, {'O2': 0}, {'Cu': 0})
    reference = copper_al2o3.oxidize_cu(replace(
        context, parameters=resolve_parameters(copper_al2o3.PARAMETERS)
    )).Node.Quantity.scaleTo(mol / (m**3 * s)).value
    # Compare the rate laws with identical parameters despite support-specific defaults.
    overrides = {
        key: spec.default for key, spec in copper_al2o3.PARAMETERS.items()
        if key.endswith('.cu_to_cuo')
    }
    actual = copper_sio2.FAMILY.kinetics_hooks['cu_sio2_oxidation_1_san_pio'](replace(
        context, parameters=resolve_parameters(copper_sio2.PARAMETERS, overrides)
    )).Node.Quantity.scaleTo(mol / (m**3 * s)).value
    assert math.isfinite(actual)
    assert actual >= 0.0
    assert actual == pytest.approx(reference, rel=1e-12, abs=0.0)
    if copper == 0.0:
        assert actual == 0.0


def check_rates(family, parameters, properties):
    from packed_bed.reactions import KineticsContext
    from packed_bed.parameters import freeze
    from daetools.pyDAE import Constant
    from pyUnits import K, Pa, mol, m, s
    class SampleModel:
        def T(self, _): return Constant(self.temperature * K)
        def P(self, _): return Constant(2e5 * Pa)
        def y_gas(self, index, _): return Constant(0.1 + index * 0.01)
        def c_sol(self, index, _): return Constant((1000 + index * 100) * mol / m**3)
        def c_gas(self, index, _): return Constant((1 + index * 0.1) * mol / m**3)
        def e_b(self, _): return Constant(0.4)
    model = SampleModel()
    for reaction in family.reactions:
        gases = {key: index for index, key in enumerate(family.required_gas_species) if key in reaction.required_species}
        solids = {key: index for index, key in enumerate(family.required_solid_species) if key in reaction.required_species}
        context = KineticsContext(model, 0, freeze(gases), freeze(solids), parameters, properties,
                                  freeze({key: key for key in (*gases, *solids)}))
        for model.temperature in (600.0, 900.0, 1200.0):
            try:
                value = family.kinetics_hooks[reaction.id](context).Node.Quantity.scaleTo(mol / (m**3 * s)).value
            except KeyError as exc:
                raise ValueError(f'{reaction.id} reads an undeclared required species or parameter: {exc}') from exc
            if not math.isfinite(value):
                raise ValueError(f'{reaction.id} has a non-finite representative rate.')


def test_all_builtin_editing_contracts_and_hidden_dependencies():
    from collections.abc import Mapping
    from dataclasses import replace
    from packed_bed.kinetics import FAMILY_REGISTRY
    from packed_bed.properties import PROPERTY_REGISTRY
    from packed_bed.reactions import ReactionDefinition
    from packed_bed.plugins.parameter_details import reaction_parameter_groups
    class ReadParameters(Mapping):
        def __init__(self, values, reads, prefix=''):
            self.values, self.reads, self.prefix = values, reads, prefix
        def __iter__(self): return iter(self.values)
        def __len__(self): return len(self.values)
        def __getitem__(self, key):
            value = self.values[key]
            name = self.prefix + key
            if isinstance(value, Mapping):
                return ReadParameters(value, self.reads, name + '.')
            self.reads.add(name)
            return value
    for name, definition in builtin_manifest().mechanisms.items():
        assert definition.parameters
        specs = definition.parameters
        family = replace(FAMILY_REGISTRY[name], reactions=tuple(ReactionDefinition(**plain(r)) for r in definition.reactions))
        groups = reaction_parameter_groups(definition)
        assert set(groups) == {reaction.id for reaction in family.reactions}
        for reaction in family.reactions:
            reads = set()
            single = replace(family, reactions=(reaction,), kinetics_hooks={reaction.id: family.kinetics_hooks[reaction.id]})
            check_rates(single, ReadParameters(resolve_parameters(specs), reads), PROPERTY_REGISTRY)
            assert reads == set(groups[reaction.id]), reaction.id
    specs = builtin_manifest().mechanisms['nickel_medrano'].parameters
    assert 'R0_M.H2' in specs and 'K0_M_PER_S.H2' in specs
    with pytest.raises(ValueError, match='not editable'):
        resolve_parameters(specs, {'REACTION_ORDER.H2': 2.0})
    with pytest.raises(ValueError, match='at least'):
        resolve_parameters(specs, {'R0_M.H2': 0.0})


@pytest.mark.parametrize('name', ['nitrogen_oxides', 'ammonia', 'enthalpy_overrides'])
def test_examples_pack_clone_and_load_without_original(tmp_path, name):
    source = tmp_path / 'source'
    shutil.copytree(EXAMPLES / name, source)
    if name != 'nitrogen_oxides':
        generate_metadata(source)
    archive = pack_plugin(source, tmp_path / 'example.msplugin')
    entry = copy_package(archive, tmp_path / 'project')
    cat = catalogue_from_project(tmp_path / 'project', [{**entry, 'enabled': True}])
    with pytest.raises(ValueError, match='before execution') if name != 'nitrogen_oxides' else __import__('contextlib').nullcontext():
        check_package(archive)
    approve_hash(code_hash(cat.paths[entry['id']]))
    assert check_package(archive) is None
    clone = tmp_path / 'clone'
    shutil.copytree(source, clone)
    data = yaml.safe_load((clone / 'manifest.yaml').read_text())
    data['id'] = 'independent_clone'
    (clone / 'manifest.yaml').write_text(yaml.safe_dump(data))
    cloned = pack_plugin(clone, tmp_path / 'cloned.msplugin')
    shutil.rmtree(source)
    shutil.rmtree(clone)
    shutil.rmtree(tmp_path / 'project')
    # Fresh subprocess, unrelated working directory, fresh local installation state.
    result = subprocess.run([sys.executable, '-c',
        'import sys; from packed_bed.plugins.storage import inspect_package; '
        'from packed_bed.plugins.check import check_package; '
        'package = inspect_package(sys.argv[1]); '
        'p = package.__enter__(); check_package(p.folder, approved=(p.digest,)); package.__exit__(None, None, None)', str(cloned)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('name', ['ammonia', 'nitrogen_oxides', 'enthalpy_overrides'])
def test_example_properties_have_consistent_numeric_and_symbolic_enthalpy(tmp_path, name):
    from daetools.pyDAE import Constant
    from pyUnits import J, K, mol
    from packed_bed.properties import PROPERTY_REGISTRY
    cat = catalogue(EXAMPLES / name, tmp_path)
    manifest = next(iter(cat.manifests.values()))
    chemistry, solids = inputs(list(manifest.species), ['Ni'], species_definitions={
        key: f'{manifest.id}:{key}' for key in manifest.species})
    environment = materialize_definitions(select_definitions(chemistry, solids, cat), cat,
                                          approved=tuple(cat.hashes.values()))
    for key, definition in manifest.species.items():
        record = environment.properties.get_record(key)
        correlation = record.enthalpy
        temperatures = np.linspace(*definition.temperature_range, 12)
        np.testing.assert_allclose((correlation.value(temperatures + .001) - correlation.value(temperatures - .001)) / .002,
                                   correlation.cp_value(temperatures), rtol=1e-7)
        assert np.all(correlation.cp_value(temperatures) > 0)
        assert np.all(record.viscosity.value(temperatures) > 0)
        for t in temperatures:
            assert correlation.dae_expression(Constant(t * K)).Node.Quantity.scaleTo(J / mol).value == pytest.approx(correlation.value(t))
            assert correlation.cp_dae_expression(Constant(t * K)).Node.Quantity.scaleTo(J / (mol * K)).value == pytest.approx(correlation.cp_value(t))
        if key in ('N2', 'CO2', 'H2O'):
            builtin = PROPERTY_REGISTRY.get_record(key)
            assert record.mw == builtin.mw and record.viscosity == builtin.viscosity
            assert correlation.value(298.15) == pytest.approx(builtin.enthalpy.value(298.15))
            assert correlation.cp_value(600) != pytest.approx(builtin.enthalpy.cp_value(600))


def test_ammonia_reaction_conserves_mass_and_uses_bindings_and_editable_rate(tmp_path):
    from daetools.pyDAE import Constant
    from pyUnits import m, mol, s
    from packed_bed.reactions import KineticsContext
    folder = tmp_path / 'ammonia'
    shutil.copytree(EXAMPLES / 'ammonia', folder)
    data = yaml.safe_load((folder / 'manifest.yaml').read_text())
    data['mechanisms']['synthesis']['values'] = {'k': .002}
    (folder / 'manifest.yaml').write_text(yaml.safe_dump(data))
    cat = catalogue(folder, tmp_path)
    gases = ['feed', 'H2', 'product']
    chemistry, solids = inputs(gases, ['Ni'], ['ammonia'], ['ammonia/synthesis'],
        species_definitions={'feed': 'builtin:N2', 'product': 'example_ammonia:NH3'},
        mechanisms={'ammonia': {'definition': 'example_ammonia:synthesis'}})
    env = materialize_definitions(select_definitions(chemistry, solids, cat), cat, approved=tuple(cat.hashes.values()))
    reaction = env.reaction_network.reactions[0]
    assert reaction.stoichiometry == {'feed': -1, 'H2': -3, 'product': 2}
    assert sum(n * env.properties.get_record(key).mw for key, n in reaction.stoichiometry.items()) == pytest.approx(0, abs=1e-12)
    assert sum(n * env.properties.enthalpy_value(key, 298.15) for key, n in reaction.stoichiometry.items()) < 0
    class Model:
        concentrations = [2, 6, 0]
        def c_gas(self, index, _):
            return Constant(self.concentrations[index] * mol / m**3)
    model = Model()
    context = KineticsContext(model, 0, dict(zip(gases, range(3))), {'Ni': 0})
    rate = lambda: env.rate_hooks[0](context).Node.Quantity.scaleTo(mol / (m**3 * s)).value
    assert rate() == pytest.approx(.002 * 2 * 6)
    for index in (0, 1):
        model.concentrations = [2, 6, 0]
        model.concentrations[index] = 0
        assert rate() == 0


def test_metadata_generation_fills_declarations_and_is_compact_and_repeatable(tmp_path):
    folder = tmp_path / 'ammonia'
    shutil.copytree(EXAMPLES / 'ammonia', folder)
    path = folder / 'manifest.yaml'
    data = yaml.safe_load(path.read_text())
    for key in ('gases', 'solids', 'reactions', 'parameters'):
        data['mechanisms']['synthesis'].pop(key)
    path.write_text(yaml.safe_dump(data))
    generated = generate_metadata(folder)
    assert generated.mechanisms['synthesis'].reactions[0].stoichiometry['NH3'] == 2
    text = path.read_text()
    assert 'maximum: null' not in text and 'catalyst_species: []' not in text
    generate_metadata(folder)
    assert path.read_text() == text
    check_package(folder, approved=(package_hash(folder),))


@pytest.mark.parametrize('name', ['../manifest.yaml', '/manifest.yaml', 'CON.txt', 'dir\\bad.py', 'name?.py'])
def test_unsafe_archives_are_rejected(tmp_path, name):
    archive = tmp_path / 'bad.msplugin'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr(name, 'bad')
    with pytest.raises(ValueError, match='path'):
        with inspect_package(archive):
            pass


def test_package_hash_ignores_zip_metadata_caches_and_rejects_tampering(tmp_path):
    folder = package(tmp_path)
    before = package_hash(folder)
    (folder / '__pycache__').mkdir()
    (folder / '__pycache__/test.pyc').write_bytes(b'cache')
    assert package_hash(folder) == before
    entry = copy_package(folder, tmp_path / 'project')
    stored = tmp_path / 'project/plugins' / entry['id'] / 'current'
    (stored / 'extra.txt').write_text('tampered')
    with pytest.raises(ValueError, match='content hash'):
        catalogue_from_project(tmp_path / 'project', [], {'builtin': '', 'plugins': {entry['id']: entry['hash']}})


def test_external_dependencies_and_cross_platform_collisions_rejected(tmp_path):
    folder = package(tmp_path)
    (folder / 'foreign.py').write_text('import non_bundled_dependency\n')
    with pytest.raises(ValueError, match='external dependencies'):
        with inspect_package(folder):
            pass
    (folder / 'foreign.py').unlink()
    (folder / 'Data').mkdir()
    (folder / 'data').mkdir()
    (folder / 'Data/a.txt').write_text('a')
    (folder / 'data/b.txt').write_text('b')
    with pytest.raises(ValueError, match='collide'):
        package_hash(folder)


def test_partial_locks_are_not_treated_as_builtin_defaults(tmp_path):
    with pytest.raises(ValueError, match='exact definition lock'):
        catalogue_from_project(tmp_path, [], {})


def test_code_approval_survives_parameter_edits_but_not_code_or_resource_changes(tmp_path):
    from packed_bed.plugins.storage import code_hash, code_approved
    folder = tmp_path / 'source'
    shutil.copytree(EXAMPLES / 'ammonia', folder)
    approve_hash(code_hash(folder))
    manifest = folder / 'manifest.yaml'
    data = yaml.safe_load(manifest.read_text())
    data['mechanisms']['synthesis']['values'] = {'k': 2e-5}
    manifest.write_text(yaml.safe_dump(data))
    assert code_approved(folder)
    assert check_package(folder) is None
    resource = folder / 'coefficients.txt'
    resource.write_text('new resource')
    assert not code_approved(folder)
    resource.unlink()
    module = folder / 'kinetics.py'
    module.write_text(module.read_text() + '\n# changed implementation\n')
    assert not code_approved(folder)


def test_relative_modules_initializers_and_resources_survive_archive_inspection(tmp_path):
    folder = tmp_path / 'source'
    shutil.copytree(EXAMPLES / 'enthalpy_overrides', folder)
    (folder / '__init__.py').write_text('from .helper import VALUE\n')
    (folder / 'helper.py').write_text('from pathlib import Path\nVALUE = int(Path(__file__).with_name("value.txt").read_text())\n')
    (folder / 'value.txt').write_text('42')
    module = folder / 'correlation.py'
    module.write_text('from . import VALUE\nassert VALUE == 42\n' + module.read_text())
    archive = pack_plugin(folder, tmp_path / 'resources.msplugin')
    digest = package_hash(folder)
    assert check_package(archive, approved=(digest,)) is None
    entry = copy_package(archive, tmp_path / 'project')
    installed = tmp_path / 'project/plugins' / entry['id'] / 'current'
    shutil.rmtree(folder)
    assert check_package(installed, approved=(digest,)) is None
