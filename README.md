\# pySAred_EH_v1.0.1


This repository contains an extended version of pySAred prepared by Dr. Elhoucine Hadden, based on the original pySAred software authored by Dr. Alexey Klechikov.


\## Attribution

Original pySAred author:

\- Dr. Alexey Klechikov

\- Original (previous) repository: https://github.com/Alexey-Klechikov/pySAred


Extended version:

\- Dr. Elhoucine Hadden

\- New repository: https://github.com/ehadden92/pySAred_EH


This extended version is published with written permission from the original author. See `NOTICE.md` and `RIGHTS_AND_PERMISSIONS.md`.



\## Main updates in this version


A detailed update summary is available here:

`docs/pySAred_EH_v1.0.1_update_summary.txt`


Main highlights include:

\- extending to support data generated at MiniAdam instrument

\- sample-shape-aware overillumination correction,

\- improved SFM consistency with batch-mode behavior,

\- sample-curvature correction reversibility fix,

\- improved numerical safety checks.

\- layout-managed GUI refactor,

\- improved small-screen usability,

\- Monitors / Time preview and export,

\- detector-image cursor/slider navigation,



## Installation and use

This repository provides:

- the Python source code: `pySAred_EH_v1.0.1.py`;
- a Windows executable: `pySAred_EH_v1.0.1.exe`, available from the GitHub Releases page.

No macOS application package is provided yet.

For detailed instructions on downloading the Windows executable, handling possible Microsoft Edge / SmartScreen warnings, verifying the SHA256 checksum, and running pySAred from the Python source code, see:

`docs/INSTALL_AND_RUN.md`


For the Windows executable, check SHA256:

`15c2a0d13c555bb713cf875e563ebfd036f0b08e92d0bde2cc4b915440ed917e`


To verify the downloaded executable on Windows:

`certutil -hashfile pySAred_EH_v1.0.1.exe SHA256`


The result should match the SHA256 hash above.



\## Rights and permissions


See:

\- `NOTICE.md`

\- `RIGHTS_AND_PERMISSIONS.md`



This repository should not be interpreted as granting a broad open-source license to the original PySAred work unless such a license is separately provided by the original author.

