# Program tab

Set the inlet and outlet conditions over time. Enter initial values first, then
add timed steps and inspect the preview.

## Program options

| Field or option | What it does |
| --- | --- |
| Flow basis — mol/s | Enters total inlet molar flow. Switching basis converts existing numerical flow values. |
| Flow basis — GHSV (h⁻¹) | Enters gas hourly space velocity using bed volume and reference conditions of 273.15 K and 100,000 Pa. Conversion needs positive bed radius and length. |
| Mode — Feed | Combines flow, temperature, and composition in one step list. |
| Mode — Independent channels | Gives inlet flow, temperature, and composition separate step lists. |
| Repeat program | Repeats channel steps over the horizon set in General. Unchecked, non-empty channels must have matching duration and determine the horizon. |

## Channels and step fields

| Field or option | What it does |
| --- | --- |
| Feed | Combined inlet flow, temperature, and mole fractions in Feed mode. |
| Inlet flow | Total molar flow or GHSV in Independent channels mode. |
| Inlet temperature (K) | Inlet gas temperature in Independent channels mode. |
| Inlet composition | Gas mole fractions in Independent channels mode. |
| Outlet pressure (Pa) | Prescribed outlet pressure, available in both modes. |
| Initial row | State at the start of the run, before any timed steps. |
| Step type — Hold | Retains the preceding state for the entered duration; no target is required. |
| Step type — Ramp | Transitions to the target over the entered duration. |
| Duration (s) | Length of this hold or ramp. |
| Target | End value for a ramp. For feed/composition, the … button opens the structured editor. |
| Channel duration | Shows the sum of the channel's step durations. |
| Focus / Show all | Gives one channel the available editing space or restores all channels. |

## Feed and composition dialog

| Field or option | What it does |
| --- | --- |
| Flow (mol/s or h⁻¹) | Initial or target inlet flow, in the active flow basis. |
| Temperature (K) | Initial or target inlet temperature. |
| Flow / Temperature checkbox on a feed ramp | Enables a new target for that part of the feed. Unticked values carry over. |
| Set composition on a feed ramp | Enables a target composition; unchecked retains the preceding composition. |
| Each gas-species field | Mole fraction of that gas. Use non-negative values whose sum is one. |
| Total | Displays the sum of the entered mole fractions and flags an unfinished or incorrect total. |
| Normalize | Scales valid non-negative fractions with a positive total to sum to one, preserving their ratios. |

## How the settings interact

Switching modes preserves each mode's inlet steps. The first visit to the other
mode copies the initial feed values and starts with no inlet steps.

A repeated cycle continues from the last cycle's final values; include a return
ramp if the cycle should end at its starting conditions. For a constant,
non-repeating run, add a hold to define its duration.

Check the preview's transitions, durations, and repeated cycles before running.

Read more: [Program settings](../GUI-GUIDE.md#program-operating-conditions-over-time) ·
[Operating program equations](../EQUATIONS.md#operating-programs).
