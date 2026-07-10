# Phase 5 xarray spike

## Decision

Adopt one `xarray.Dataset` and one `results.nc` file as the runtime result
representation. The spike materially removed custom axis, pandas MultiIndex,
and plotting-extraction code rather than adding another representation.

## Cases exercised

- A DAETools inert N2/Ni run with three cells, four faces, two reported times,
  and integral heat/mass variables.
- A synthetic realistic shape with five times, six cells, seven faces, four gas
  species, three solid species, and two reactions.

The realistic dataset made a SciPy-engine NetCDF round trip with string species
and reaction coordinates, units metadata, and derived solid mole fractions
intact. The file was 6,152 bytes. Solid mole fractions summed to one within
floating-point tolerance.

## Comparison

| Concern | pandas/MultiIndex path | xarray path |
| --- | --- | --- |
| Extraction | Separate dataframe and plotting extractors | One `extract_dataset` |
| Axes | Custom flattening and MultiIndex construction | Named dimensions and coordinates |
| Cells/faces | Encoded into column levels | Separate `x_cell` and `x_face` dimensions |
| Species/reactions | Repeated column-label logic | Coordinates on `gas_species`, `solid_species`, and `reaction` |
| Units | Implicit in column names | Per-variable attributes |
| Durable output | Several Python-specific pickle files | One portable `results.nc` |
| Plotting | Re-read and reshape the DAETools process | Direct labelled selection |
| Balances | Separate dataframe builders | Requested solver totals plus derived error DataArrays |
| ML conversion | Fixed, repository-specific 42-column script | Separate explicit variable selection and stacking |

For the spike, stacking temperature and pressure over time produced a labelled
`(5, 12)` matrix without custom column construction.

## Cost

xarray is imported only during extraction, loading, plotting, or downstream
conversion. Pure configuration validation does not import xarray or DAETools.
SciPy is used as the NetCDF engine because it is already an installed numerical
dependency; `netCDF4` is not required.
