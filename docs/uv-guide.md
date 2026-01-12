# Minimal uv Guide for Your Project

## 1️⃣ Sync the environment

```bash
uv sync
```

**What it does:**

- Creates or updates the virtual environment (`.venv`)
- Installs all dependencies from `pyproject.toml`
- Freezes exact versions in `uv.lock`
- Removes packages not listed in `pyproject.toml`

> ✅ Always run this after adding or updating dependencies.

---

## 2️⃣ Install the project in editable mode

```bash
uv pip install -e .
```

**Why you need this:**

- Makes your `src/` folder importable from anywhere:

    ```python
    from src.core.config import RAW_DATA_DIR
    ```

- Keeps it editable — any changes in `src/` are immediately available without reinstalling

---

## 3️⃣ Recommended workflow

```bash
# 1. Sync dependencies and environment
uv sync

# 2. Install the project so `src` is importable
uv pip install -e .

# 3. Launch Jupyter or run scripts
uv run jupyter lab
uv run python src/models/train.py
```

> After this, you are ready to run scripts or notebooks with the correct environment.

---

## ✅ Notes

- Only `pyproject.toml` and `uv.lock` define your environment. `requirements.txt` is ignored.
- Restart Jupyter kernels after running `uv sync` or `uv pip install -e .` to pick up changes.
