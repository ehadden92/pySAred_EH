# Install and run pySAred_EH_v1.0.1

## 1. Available options

Users can run pySAred_EH_v1.0.1 in two ways:

1. Download and run the Windows executable.
2. Run the Python source code using a Python environment.

At the moment, no macOS application package is provided yet. 
Until then, we kindly ask macOS or Linux users to refer to the instructions below on how to use the GUI via the provided source code.

## 2. Windows executable

The Windows executable is available from the GitHub Releases page:

`pySAred_EH_v1.0.1.exe`

Official release page:

`https://github.com/ehadden92/pySAred_EH/releases`

## 3. Browser / SmartScreen warning during download

Because the executable is not yet code-signed, some browsers (e.g. Microsoft Edge) or Windows SmartScreen may show a warning such as:

> Make sure you trust `pySAred_EH_v1.0.1.exe` before you open it.  
> Microsoft Defender SmartScreen couldn't verify if this file is safe because it isn't commonly downloaded.  
> Publisher: Unknown

This warning does not necessarily mean the file is malware. It means the file is new, not widely downloaded yet, and not signed by a verified publisher.

Users should make sure that the file is downloaded from the official GitHub release page.

Users who trust the source may choose to keep/download the file. They should then verify the SHA256 checksum before opening it.

## 4. Verify the downloaded executable

Expected SHA256:

`15c2a0d13c555bb713cf875e563ebfd036f0b08e92d0bde2cc4b915440ed917e`

To verify the downloaded executable on Windows, open Command Prompt or PowerShell in the download folder and run:

`certutil -hashfile pySAred_EH_v1.0.1.exe SHA256`

The result should match the SHA256 hash above. This means the downloaded file is identical to the official release file.

If the hash does not match, do not run the file. Delete it and download it again from the official GitHub release page.


## 5. First launch on Windows

After verifying the SHA256 hash, double-click:

`pySAred_EH_v1.0.1.exe`

Because the executable is not yet code-signed, Windows may show another SmartScreen warning when the application is opened for the first time.

Depending on the Windows version and security settings, users may need to click:

`More info` → `Run anyway`

or equivalent wording.

This is expected for a new unsigned executable. It does not necessarily mean that the file is malware.

Users who trust the source and have verified the SHA256 hash may also choose:

`Report this app as safe`

in Browser / SmartScreen. This may help future downloads build reputation.

## 6. Build notes for the Windows executable

The Windows executable was:

- built on Windows from a clean conda environment,
- packaged with PyInstaller using `--noupx`,
- checked while Microsoft Defender real-time protection was enabled,
- scanned using a targeted Microsoft Defender custom scan without error.

## 7. Running pySAred from the Python source code

Users who do not want to use the Windows executable, or users on macOS/Linux, can run pySAred from the provided Python source code.

The source file is:

`pySAred_EH_v1.0.1.py`

The required Python packages are listed in:

`requirements.txt`

The easiest recommended method is to use Anaconda or Miniconda.

## 8. Running from source code on Windows using Anaconda PowerShell Prompt

### Step 1 — Install Anaconda or Miniconda

If Anaconda or Miniconda is not already installed, install one of them first.

After installation, open:

`Anaconda PowerShell Prompt`

This is different from normal PowerShell because it already knows where `conda` is installed.

### Step 2 — Download the source code

Go to the GitHub repository:

`https://github.com/ehadden92/pySAred_EH`

Go to the latest release and download the source code: `Source code (zip)`

Then unzip the downloaded file. For example, after unzipping, the folder may be located in:

`C:\Users\YOUR_USERNAME\Downloads\pySAred_EH-1.0.1-EH`


### Step 3 — Go to the source-code directory

In Anaconda PowerShell Prompt, move to the folder where the source code was extracted.

Example:

```powershell
cd "$env:USERPROFILE\Downloads\pySAred_EH-1.0.1-EH"
```

If your folder is somewhere else, replace the path with your actual folder path.

To check that you are in the correct folder, run:

```powershell
dir
```

You should see files such as:

```text
pySAred_EH_v1.0.1.py
requirements.txt
README.md
```

### Step 4 — Create a clean conda environment

Run:

```powershell
conda create -n pysared_env python=3.10 -y
```

This creates a new Python environment named:

`pysared_env`

### Step 5 — Activate the environment

Run:

```powershell
conda activate pysared_env
```

After this, the beginning of the prompt should show:

```text
(pysared_env)
```

### Step 6 — Install the required packages

Run:

```powershell
conda install -c conda-forge pyqt=5 numpy scipy h5py matplotlib pyqtgraph -y
```

This may take a few minutes.

### Step 7 — Start pySAred

Run:

```powershell
python pySAred_EH_v1.0.1.py
```

The pySAred graphical interface should open. Please note it may take from few seconds up to a couple of minutes depending on the used device to open for the first time.

## 9. Running from source code on Windows using normal PowerShell

This method is only recommended if Python is already installed on the computer.

If you are not sure whether Python is installed, use the Anaconda method above instead.

### Step 1 — Open PowerShell

Open the Windows Start menu and search for:

`PowerShell`

Then open it.

### Step 2 — Go to the source-code folder

Example:

```powershell
cd "$env:USERPROFILE\Downloads\pySAred_EH-1.0.1-EH"
```

If your folder is somewhere else, replace the path with your actual folder path.

Check the folder contents:

```powershell
dir
```

You should see:

```text
pySAred_EH_v1.0.1.py
requirements.txt
README.md
```

### Step 3 — Create a Python virtual environment

Run:

```powershell
python -m venv pysared_env
```

### Step 4 — Activate the environment

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

### Step 5 — Install the required packages

Run:

```powershell
python -m pip install --upgrade pip
```

Then run:

```powershell
python -m pip install -r requirements.txt
```

### Step 6 — Start pySAred

Run:

```powershell
python pySAred_EH_v1.0.1.py
```

The pySAred graphical interface should open.

## 10. Running from source code on macOS or Linux

No macOS application package is provided yet.

macOS and Linux users should run pySAred from the Python source code.

The easiest method is also to use conda.

### Step 1 — Open Terminal

Open the Terminal application.

### Step 2 — Download and unzip the source code

Go to the GitHub repository:

`https://github.com/ehadden92/pySAred_EH`

Go to the latest release and download the source code: `Source code (zip)`

Then unzip the downloaded file. For example, after unzipping, the folder may be located in:

```text
~/Downloads/pySAred_EH-1.0.1-EH
```

### Step 3 — Go to the source-code folder

In Terminal, run:

```bash
cd ~/Downloads/pySAred_EH-1.0.1-EH
```

If your folder is somewhere else, replace the path with your actual folder path.

Check the folder contents:

```bash
ls
```

You should see:

```text
pySAred_EH_v1.0.1.py
requirements.txt
README.md
```

### Step 4 — Create a clean conda environment

Run:

```bash
conda create -n pysared_env python=3.10 -y
```

### Step 5 — Activate the environment

Run:

```bash
conda activate pysared_env
```

After this, the beginning of the prompt should show:

```text
(pysared_env)
```

### Step 6 — Install the required packages

Run:

```bash
conda install -c conda-forge pyqt=5 numpy scipy h5py matplotlib pyqtgraph -y
```

This may take a few minutes.

### Step 7 — Start pySAred

Run:

```bash
python pySAred_EH_v1.0.1.py
```

The pySAred graphical interface should open. 
Please note it may take from few seconds up to a couple of minutes depending on the used device to open for the first time.

## 11. Common issues

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

Then run:

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

This can happen because the executable is not yet code-signed.

Users should:

1. make sure the file was downloaded from the official GitHub release page,
2. verify the SHA256 hash,
3. if they trust the source, choose `More info` → `Run anyway` or equivalent wording.

## 12. Summary

Recommended for most Windows users:

1. Download `pySAred_EH_v1.0.1.exe` from the GitHub Releases page.
2. Verify the SHA256 hash.
3. Run the executable.
4. If SmartScreen appears, use `More info` → `Run anyway` if the source and hash are trusted.

Recommended for macOS/Linux users, or users who prefer source code:

1. Download the repository source code.
2. Create a clean conda environment.
3. Install the required packages.
4. Run:

```bash
python pySAred_EH_v1.0.1.py
```
