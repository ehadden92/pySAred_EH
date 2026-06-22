# Changelog

## pySAred_EH v1.1.0

Main changes compared to pySAred_EH v1.0.1:

- Added support for SuperADAM Nomad NeXus `.nxs` files.
- Preserved the legacy `.h5` workflow.
- Added dynamic Nomad scanned-variable unpacking using the labels stored in the file.
- Added configurable NR / NSF / PNR detector-frame unpacking for Nomad files.
- Applied the selected Nomad unpacking consistently to detector data and frame-indexed scanned variables.
- Added `Monitor1` support for Nomad monitor normalization, with fallback warning behavior.
- Improved Nomad ROI and slit handling.
- Added a separate data-reduction summary file during export.
- Improved the Nomad unpacking dialog readability and small-screen usability.
- Updated repository documentation to distinguish `v1.0.1` legacy/pre-Nomad usage from `v1.1.0` Nomad-compatible usage.

## pySAred_EH v1.0.1

Main changes compared to the original pySAred version by Dr. Alexey Klechikov are documented in:

`docs/pySAred_EH_v1.0.1_update_summary.txt`
