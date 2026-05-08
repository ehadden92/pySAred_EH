\# pySAred\_EH\_v1.0.1



This release provides a Windows executable for the extended pySAred version prepared by Dr. Elhoucine Hadden.



Original pySAred author:

\- Dr. Alexey Klechikov

\- Original repository: https://github.com/Alexey-Klechikov/pySAred



Extended version:

\- Dr. Elhoucine Hadden

\- New repository: https://github.com/ehadden92/pySAred\_EH



This extended version is published based on written permission from the original author, with attribution preserved. See `NOTICE.md` and `RIGHTS\_AND\_PERMISSIONS.md` in the repository.



\## Main update highlights



\- Layout-managed GUI refactor.

\- Improved small-screen usability.

\- Added Monitors / Time preview and export workflow.

\- Added detector-image cursor/slider navigation.

\- Extended overillumination correction to selected non-rectangular symmetric sample shapes.

\- Restored reversibility of sample-curvature correction.

\- Applied SFM curvature correction consistently to signal and background ROIs.

\- Aligned SFM projected-FWHM / resolution behavior with batch mode.

\- Improved numerical validation and plotting robustness.



A detailed technical summary is available in:



`docs/pySAred\_EH\_v1.0.1\_update\_summary.txt`



\## Download



Recommended Windows executable:



`pySAred\_EH\_v1.0.1.exe`



\## SHA256 checksum



`15c2a0d13c555bb713cf875e563ebfd036f0b08e92d0bde2cc4b915440ed917e`



To verify after downloading on Windows:



`certutil -hashfile pySAred\_EH\_v1.0.1.exe SHA256`



The output should match the SHA256 hash above.



\## Build notes



\- Built on Windows from a clean conda environment: `pysared\_build\_101`.

\- Packaged with PyInstaller using `--noupx`.

\- Microsoft Defender real-time protection was enabled.

\- Targeted Microsoft Defender custom scan completed without error.



\## Rights and permissions



This release preserves attribution to the original author. It should not be interpreted as granting a broad open-source license to the original PySAred work unless such a license is separately provided by the original author.



See:



\- `NOTICE.md`

\- `RIGHTS\_AND\_PERMISSIONS.md`



\## Note for Windows users



This executable is not yet code-signed. Some Windows systems may display a SmartScreen or unknown-publisher warning for newly downloaded executables. Future releases may improve this through code signing.

