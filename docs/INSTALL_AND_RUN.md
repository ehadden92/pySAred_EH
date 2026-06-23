# Install and run pySAred_EH

This document explains how to install and run the available pySAred_EH versions.

The repository currently provides:

- `pySAred_EH v1.0.1`, the first public pySAred_EH extension compared to the original pySAred code.
- `pySAred_EH v1.1.0`, the newer version adding SuperADAM Nomad NeXus `.nxs` support while preserving the legacy `.h5` workflow.

Official repository:

`https://github.com/ehadden92/pySAred_EH`

Official releases page:

`https://github.com/ehadden92/pySAred_EH/releases`

## 1. Which version should I use?

| Version | Recommended use | Data formats |
|---|---|---|
| `v1.1.0` | Current SuperADAM users, especially data acquired with the Nomad interface | Nomad `.nxs`, legacy `.h5` |
| `v1.0.1` | Reproducing or comparing reductions performed with the earlier public pre-Nomad release | legacy `.h5` |

For new SuperADAM Nomad data, use `v1.1.0`.

For older pre-Nomad reductions, we also recommend the use of the new version which includes multiple fixes compared to the legacy code.

## 2. Available options

Users can run pySAred_EH in two ways:

1. Download and run the Windows executable package from the GitHub Releases page.
2. Run the Python source code using a Python environment.

At the moment, no macOS application package is provided. macOS and Linux users should run the GUI from the provided Python source code.

## 3. Windows executable packages

### pySAred_EH v1.1.0

Recommended Windows release asset:

`pySAred_EH_v1.1.0_Windows_x64.zip`

This ZIP package contains the Windows executable and all files required by the PyInstaller build. Extract the full ZIP folder before running the program.

Do not move the `.exe` file outside the extracted folder, because it needs the included support files.

After extraction, run:

`pySAred_EH_v110.exe`

### pySAred_EH v1.0.1

Legacy Windows executable:

`pySAred_EH_v1.0.1.exe`

This executable is kept for users who need the previous public pre-Nomad release.

## 4. Browser / Windows SmartScreen warning

The Windows executable packages are not code-signed. Some browsers, Microsoft Defender SmartScreen, or Windows security settings may therefore show warnings such as:

> Make sure you trust this file before you open it.  
> Microsoft Defender SmartScreen couldn't verify if this file is safe because it isn't commonly downloaded.  
> Publisher: Unknown

This warning does not necessarily mean that the file is malware. It usually means that the file is new, not widely downloaded yet, and not signed by a verified publisher.

Users should:

1. download files only from the official GitHub Releases page,
2. verify the SHA256 checksum,
3. run the file only if they trust the source and the checksum matches.

## 5. Verify the downloaded files

### Recommended verification for v1.1.0

For `v1.1.0`, verify the downloaded ZIP file:

`pySAred_EH_v1.1.0_Windows_x64.zip`

The expected SHA256 checksum should be taken is:

`4AFD609959D9DAA0699FE037C89440A61C9B9B2CF8A5FE330BCC6ACF40E5DE45`

To verify the ZIP on Windows, open Command Prompt or PowerShell in the download folder and run:

```powershell
certutil -hashfile pySAred_EH_v1.1.0_Windows_x64.zip SHA256
```

The result should match the SHA256 checksum published in the official release.

If the hash does not match, do not run the program. Delete the file and download it again from the official GitHub Releases page.

### Verification for v1.0.1

For `v1.0.1`, verify the legacy executable:

`pySAred_EH_v1.0.1.exe`

Expected SHA256:

`15c2a0d13c555bb713cf875e563ebfd036f0b08e92d0bde2cc4b915440ed917e`

To verify it on Windows, open Command Prompt or PowerShell in the download folder and run:

```powershell
certutil -hashfile pySAred_EH_v1.0.1.exe SHA256
```

The result should match the SHA256 hash above.

## 6. First launch on Windows

### v1.1.0

1. Download `pySAred_EH_v1.1.0_Windows_x64.zip` from the official GitHub Release.
2. Verify the SHA256 checksum.
3. Extract the complete ZIP folder.
4. Open the extracted folder.
5. Double-click:

`pySAred_EH_v110.exe`

If Windows SmartScreen appears, users who trust the source and have verified the checksum may need to click:

`More info` -> `Run anyway`

or equivalent wording.

### v1.0.1

1. Download `pySAred_EH_v1.0.1.exe` from the official GitHub Release.
2. Verify the SHA256 checksum.
3. Double-click:

`pySAred_EH_v1.0.1.exe`

If Windows SmartScreen appears, users who trust the source and have verified the checksum may need to click:

`More info` -> `Run anyway`

or equivalent wording.

## 7. Build notes for the Windows executable packages

The Windows executable packages are built on Windows from a conda environment and packaged with PyInstaller.

For the `v1.1.0` Windows package, the recommended release asset is a PyInstaller one-directory build distributed as a ZIP file. This is preferred over a single-file executable because it is easier to inspect, easier to debug, and usually more robust for a PyQt / NumPy / SciPy / h5py / Matplotlib application.

Recommended build-trust practices include:

- building from a clean or dedicated conda environment,
- packaging with PyInstaller using `--noupx`,
- keeping the ZIP structure intact,
- publishing `SHA256SUMS.txt`,
- publishing build-environment information,
- scanning the built package with Microsoft Defender before release.

## 8. Running pySAred from Python source code

Users who do not want to use the Windows executable package, or users on macOS/Linux, can run pySAred_EH from the provided Python source code.

Source-code files:

- `pySAred_EH_v110.py` for `v1.1.0`
- `pySAred_EH_v1.0.1.py` for `v1.0.1`

The required Python packages are listed in:

`requirements.txt`

The easiest recommended method is to use Anaconda or Miniconda.

## 9. Running from source code on Windows using Anaconda PowerShell Prompt

### Step 1 - Install Anaconda or Miniconda

If Anaconda or Miniconda is not already installed, install one of them first.

After installation, open:

`Anaconda PowerShell Prompt`

This is different from normal PowerShell because it already knows where `conda` is installed.

### Step 2 - Download the source code

Go to the GitHub Releases page:

`https://github.com/ehadden92/pySAred_EH/releases`

Download the source code for the version you want to use.

For `v1.1.0`, use the `v1.1.0` release.

For `v1.0.1`, use the `v1.0.1` release.

Unzip the downloaded source code.

### Step 3 - Go to the source-code directory

In Anaconda PowerShell Prompt, move to the folder where the source code was extracted.

Example for `v1.1.0`:

```powershell
cd "$env:USERPROFILE\Downloads\pySAred_EH-1.1.0"
```

If your folder is somewhere else, replace the path with your actual folder path.

Check the folder contents:

```powershell
dir
```

For `v1.1.0`, you should see files such as:

```text
pySAred_EH_v110.py
requirements.txt
README.md
```

For `v1.0.1`, you should see files such as:

```text
pySAred_EH_v1.0.1.py
requirements.txt
README.md
```

### Step 4 - Create a clean conda environment

Run:

```powershell
conda create -n pysared_env python=3.10 -y
```

This creates a new Python environment named:

`pysared_env`

### Step 5 - Activate the environment

Run:

```powershell
conda activate pysared_env
```

After this, the beginning of the prompt should show:

```text
(pysared_env)
```

### Step 6 - Install the required packages

Run:

```powershell
conda install -c conda-forge pyqt=5 numpy scipy h5py matplotlib pyqtgraph -y
```

This may take a few minutes.

### Step 7 - Start pySAred

For `v1.1.0`, run:

```powershell
python pySAred_EH_v110.py
```

For `v1.0.1`, run:

```powershell
python pySAred_EH_v1.0.1.py
```

The pySAred graphical interface should open. It may take from a few seconds up to a couple of minutes depending on the device, especially on the first launch.

## 10. Running from source code on Windows using normal PowerShell

This method is only recommended if Python is already installed on the computer.

If you are not sure whether Python is installed, use the Anaconda method above instead.

### Step 1 - Open PowerShell

Open the Windows Start menu and search for:

`PowerShell`

Then open it.

### Step 2 - Go to the source-code folder

Example for `v1.1.0`:

```powershell
cd "$env:USERPROFILE\Downloads\pySAred_EH-1.1.0"
```

If your folder is somewhere else, replace the path with your actual folder path.

Check the folder contents:

```powershell
dir
```

### Step 3 - Create a Python virtual environment

Run:

```powershell
python -m venv pysared_env
```

### Step 4 - Activate the environment

Run:

```powershell
.\pysared_env\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then try again:

```powershell
.\pysared_env\Scripts\Activate.ps1
```

After activation, the beginning of the prompt should show:

```text
(pysared_env)
```

### Step 5 - Install the required packages

Run:

```powershell
python -m pip install --upgrade pip
```

Then run:

```powershell
python -m pip install -r requirements.txt
```

### Step 6 - Start pySAred

For `v1.1.0`, run:

```powershell
python pySAred_EH_v110.py
```

For `v1.0.1`, run:

```powershell
python pySAred_EH_v1.0.1.py
```

The pySAred graphical interface should open.

## 11. Running from source code on macOS or Linux

No macOS application package is provided yet.

macOS and Linux users should run pySAred_EH from the Python source code.

The easiest method is also to use conda.

### Step 1 - Open Terminal

Open the Terminal application.

### Step 2 - Download and unzip the source code

Go to the GitHub Releases page:

`https://github.com/ehadden92/pySAred_EH/releases`

Download the source code for the version you want to use.

Unzip the downloaded source code.

### Step 3 - Go to the source-code folder

Example for `v1.1.0`:

```bash
cd ~/Downloads/pySAred_EH-1.1.0
```

If your folder is somewhere else, replace the path with your actual folder path.

Check the folder contents:

```bash
ls
```

### Step 4 - Create a clean conda environment

Run:

```bash
conda create -n pysared_env python=3.10 -y
```

### Step 5 - Activate the environment

Run:

```bash
conda activate pysared_env
```

After this, the beginning of the prompt should show:

```text
(pysared_env)
```

### Step 6 - Install the required packages

Run:

```bash
conda install -c conda-forge pyqt=5 numpy scipy h5py matplotlib pyqtgraph -y
```

This may take a few minutes.

### Step 7 - Start pySAred

For `v1.1.0`, run:

```bash
python pySAred_EH_v110.py
```

For `v1.0.1`, run:

```bash
python pySAred_EH_v1.0.1.py
```

The pySAred graphical interface should open. It may take from a few seconds up to a couple of minutes depending on the device, especially on the first launch.

## 12. Common issues

### `conda` is not recognized

This usually means Anaconda or Miniconda is not installed, or the wrong terminal was opened.

On Windows, use:

`Anaconda PowerShell Prompt`

instead of normal PowerShell.

### `python` is not recognized

Python is not installed, or it is not available in the current terminal.

The simplest solution is to install Anaconda or Miniconda and then use:

`Anaconda PowerShell Prompt`

### The graphical interface does not open

Make sure the environment is activated.

For conda users, run:

```powershell
conda activate pysared_env
```

Then run the appropriate source file:

```powershell
python pySAred_EH_v110.py
```

or:

```powershell
python pySAred_EH_v1.0.1.py
```

### A package is missing

If an error says that a module is missing, reinstall the required packages.

For conda users, run:

```powershell
conda install -c conda-forge pyqt=5 numpy scipy h5py matplotlib pyqtgraph -y
```

For pip users, run:

```powershell
python -m pip install -r requirements.txt
```

### PowerShell blocks activation of the virtual environment

If this command fails:

```powershell
.\pysared_env\Scripts\Activate.ps1
```

run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then try again:

```powershell
.\pysared_env\Scripts\Activate.ps1
```

This changes the execution policy only for the current PowerShell window.

### The Windows executable is blocked by SmartScreen

This can happen because the executable is not code-signed.

Users should:

1. make sure the file was downloaded from the official GitHub Releases page,
2. verify the SHA256 checksum,
3. if they trust the source, choose `More info` -> `Run anyway` or equivalent wording.

## 13. Summary

Recommended for most current Windows users:

1. Download `pySAred_EH_v1.1.0_Windows_x64.zip` from the GitHub Releases page.
2. Verify the SHA256 checksum published in the release notes or in `SHA256SUMS.txt`.
3. Extract the complete ZIP folder.
4. Run `pySAred_EH_v110.exe`.
5. If SmartScreen appears, use `More info` -> `Run anyway` only if the source and checksum are trusted.

Recommended for users reproducing pre-Nomad reductions:

1. Download `pySAred_EH_v1.0.1.exe` from the GitHub Releases page.
2. Verify the SHA256 hash.
3. Run the executable.

Recommended for macOS/Linux users, or users who prefer source code:

1. Download the repository source code for the desired release.
2. Create a clean conda environment.
3. Install the required packages.
4. Run the appropriate source file:

```bash
python pySAred_EH_v110.py
```

or:

```bash
python pySAred_EH_v1.0.1.py
```
