# Zoroaster

A small Python project that reads Wikipedia page data via Wikipedia's free
[REST API](https://en.wikipedia.org/api/rest_v1/) (no API key required).

`main.py` fetches a Wikipedia page and prints its summary.

## Setup

Requires Python 3.10+.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Command Prompt (`cmd.exe`):

```bat
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

To leave the virtual environment later, run `deactivate`.

## Usage

```bash
# Default page (Zoroastrianism)
python main.py

# A specific page
python main.py "Albert Einstein"

# A different language edition
python main.py "Albert Einstein" --lang de
```

## Dependencies

- [`requests`](https://pypi.org/project/requests/) — HTTP client used to call
  the Wikipedia REST API.
