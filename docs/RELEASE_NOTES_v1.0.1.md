\# pySAred_EH_v1.0.1


This release provides a Windows executable for the extended pySAred version prepared by Dr. Elhoucine Hadden.

Original pySAred author:

\- Dr. Alexey Klechikov

\- Original (previous) repository: https://github.com/Alexey-Klechikov/pySAred

Extended version:

\- Dr. Elhoucine Hadden

\- New repository: https://github.com/ehadden92/pySAred_EH


This extended version is published based on written permission from the original author, with attribution preserved. See `NOTICE.md` and `RIGHTS_AND_PERMISSIONS.md` in the repository.


\## Main update highlights

\- extending to support data generated at MiniAdam instrument,

\- Extended overillumination correction to selected non-rectangular symmetric sample shapes.

\- Restored reversibility of sample-curvature correction.

\- Applied SFM curvature correction consistently to signal and background ROIs.

\- Aligned SFM projected-FWHM / resolution behavior with batch mode.

\- Added Monitors / Time / slits preview and export workflow.

\- Layout-managed GUI refactor.

\- Improved small-screen usability.

\- Added detector-image cursor/slider navigation.

\- Improved numerical validation and plotting robustness.


A detailed technical summary is available in:

`docs/pySAred_EH_v1.0.1_update_summary.txt`


\# Windows executable:

`pySAred_EH_v1.0.1.exe`


\## SHA256 checksum

`15c2a0d13c555bb713cf875e563ebfd036f0b08e92d0bde2cc4b915440ed917e`

To verify after downloading on Windows:

`certutil -hashfile pySAred_EH_v1.0.1.exe SHA256`

The output should match the SHA256 hash above.


No macOS application package is provided yet. 
The current alternative for macOS or Linux users is to run directly from the source code (detailed instructions provided).


\## Build notes

\- Built on Windows from a clean conda environment: `pysared_build_101`.

\- Packaged with PyInstaller using `--noupx`.

\- Microsoft Defender real-time protection was enabled.

\- Targeted Microsoft Defender custom scan completed without error.


\## Rights and permissions

This release preserves attribution to the original author. It should not be interpreted as granting a broad open-source license to the original PySAred work unless such a license is separately provided by the original author.

See:

\- `NOTICE.md`

\- `RIGHTS_AND_PERMISSIONS.md`


\## Note for Windows users

This executable is not yet code-signed. Some Windows systems may display a SmartScreen or unknown-publisher warning for newly downloaded executables. Future releases may improve this through code signing.

