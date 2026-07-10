from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from packed_bed.kinetics import FAMILY_REGISTRY, load_reaction_families
from packed_bed.reactions import build_reaction_network, reaction_catalog


NICKEL_FAMILY = FAMILY_REGISTRY["nickel_medrano"]
REFORMING_FAMILY = FAMILY_REGISTRY["reforming_xu_froment"]


def test_explicit_family_registry_is_import_safe() -> None:
    script = """
import builtins
import sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'daetools' or name.startswith('daetools.') or name == 'pyUnits':
        raise AssertionError(f'forbidden solver import: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from packed_bed.kinetics import FAMILY_REGISTRY
assert tuple(FAMILY_REGISTRY) == (
    'nickel_medrano',
    'reforming_xu_froment',
    'reforming_numaguchi',
    'copper_san_pio',
    'iron_he',
)
"""
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
    )


def test_family_loader_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown reaction families: unknown"):
        load_reaction_families(("unknown",))


def test_family_requirements_are_local_to_the_mechanism() -> None:
    assert NICKEL_FAMILY.required_solid_species == ("Ni", "NiO")
    assert REFORMING_FAMILY.required_solid_species == ("Ni",)
    assert not ({"Al2O3", "CuAl2O4", "CuAlO2"} & set(NICKEL_FAMILY.required_solid_species))


def test_catalog_contains_only_selected_families() -> None:
    catalog = reaction_catalog((NICKEL_FAMILY,))

    assert set(catalog) == {
        "ni_reduction_h2_medrano",
        "ni_reduction_co_medrano",
        "ni_oxidation_o2_medrano",
    }
    assert "smr_reaction_xu_froment" not in catalog


def test_rejects_missing_stoichiometric_species() -> None:
    with pytest.raises(ValueError, match="requires unselected species: H2O"):
        build_reaction_network(
            ("ni_reduction_h2_medrano",),
            ("H2",),
            ("Ni", "NiO"),
            families=(NICKEL_FAMILY,),
        )


def test_rejects_missing_catalyst_species() -> None:
    with pytest.raises(ValueError, match="requires unselected species: Ni"):
        build_reaction_network(
            ("smr_reaction_xu_froment",),
            ("CH4", "H2O", "CO", "H2"),
            (),
            families=(REFORMING_FAMILY,),
        )


def test_accepts_selected_solid_catalyst_for_gas_reaction() -> None:
    network = build_reaction_network(
        ("smr_reaction_xu_froment",),
        ("CH4", "H2O", "CO", "H2"),
        ("Ni",),
        families=(REFORMING_FAMILY,),
    )

    assert network.reaction_ids == ("smr_reaction_xu_froment",)
