# src/ml_in_finance_ensae/data.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import zipfile
import pandas as pd

@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path = Path("data/raw")


FF25_ZIP = "25_Portfolios_5x5_CSV.zip"
FF3_ZIP = "F-F_Research_Data_Factors_CSV.zip"


def _read_first_csv_from_zip(zip_path: Path) -> list[str]:
    """Return file lines (decoded) from the first .csv found in the zip."""
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing file: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found in {zip_path.name}. Contains: {zf.namelist()}")
        with zf.open(csv_names[0]) as f:
            # latin1 is usually safe for French files
            text = f.read().decode("latin1", errors="replace")
    return text.splitlines()


def _parse_monthly_table(lines: list[str]) -> pd.DataFrame:
    """
    Parse the first monthly table in a French CSV (often contains metadata + multiple tables).
    We locate the first line that looks like data (YYYYMM,...) and use the closest header above it.
    """
    cleaned = [ln.strip() for ln in lines if ln.strip()]
    header = None
    rows: list[list[str]] = []

    def is_yyyymm_row(s: str) -> bool:
        # data rows start with YYYYMM then comma
        if "," not in s:
            return False
        first = s.split(",")[0].strip()
        return first.isdigit() and len(first) == 6

    for i, line in enumerate(cleaned):
        if header is None:
            # Try direct header detection
            norm = line.replace(" ", "").lower()
            if norm.startswith("date,"):
                header = [h.strip() for h in line.split(",")]
                continue

            # Otherwise, if this line looks like data (YYYYMM,...), infer header from previous line
            if is_yyyymm_row(line):
                if i == 0:
                    raise ValueError("Found data row at top of file; cannot infer header.")
                header = [h.strip() for h in cleaned[i - 1].split(",")]
                # Now treat this line as first data row (fall through below)
            else:
                continue

        # Once we have a header, collect data rows until table ends
        parts = [p.strip() for p in line.split(",")]

        if len(parts) != len(header):
            break  # end of table

        if not is_yyyymm_row(line):
            break  # end of table (non YYYYMM)

        rows.append(parts)

    if header is None or not rows:
        # helpful debug: show first lines
        preview = "\n".join(cleaned[:30])
        raise ValueError(
            "Could not locate a monthly table. First lines preview:\n" + preview
        )

    df = pd.DataFrame(rows, columns=header)

    # First column is date (whatever its name)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
    df = df.set_index("Date").sort_index()

    # Convert values to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace common sentinels
    sentinels = [-99.99, -999.0, 99.99, 999.0, -9999.0, 9999.0]
    df = df.replace(sentinels, pd.NA)

    # Percent -> decimal
    df = df / 100.0

    # Safety: drop duplicate dates
    df = df[~df.index.duplicated(keep="first")].sort_index()

    return df


def load_ff25_and_rf(paths: DataPaths = DataPaths()) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      ff25_excess: (T x 25) excess returns of the 25 size-B/M portfolios (decimal, monthly)
      rf:          (T,) risk-free rate (decimal, monthly)
    """
    ff25_zip_path = paths.raw_dir / FF25_ZIP
    ff3_zip_path = paths.raw_dir / FF3_ZIP

    ff25_lines = _read_first_csv_from_zip(ff25_zip_path)
    ff3_lines = _read_first_csv_from_zip(ff3_zip_path)

    ff25 = _parse_monthly_table(ff25_lines)
    ff3 = _parse_monthly_table(ff3_lines)

    # RF column check
    if "RF" not in ff3.columns:
        raise ValueError(f"RF column not found. Columns are: {list(ff3.columns)}")

    rf = ff3["RF"].copy()
    rf.name = "RF"

    # Align by common dates
    common = ff25.index.intersection(rf.index)
    ff25 = ff25.loc[common]
    rf = rf.loc[common]

    # Excess returns
    ff25_excess = ff25.sub(rf, axis=0)

    return ff25_excess, rf

FF3_ZIP = "F-F_Research_Data_Factors_CSV.zip"

def load_ff3_factors(paths: "DataPaths" = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load monthly Fama-French 3 factors (Mkt-RF, SMB, HML) and RF from local ZIP.
    Returns:
      factors: DataFrame with columns ["Mkt-RF", "SMB", "HML"] in decimals
      rf: Series in decimals
    Assumes helpers exist:
      - _read_first_csv_from_zip(zip_path) -> list[str]
      - _parse_monthly_table(lines) -> pd.DataFrame (with dates index)
    """
    if paths is None:
        paths = DataPaths()

    zip_path = Path(paths.raw_dir) / FF3_ZIP
    if not zip_path.exists():
        raise FileNotFoundError(f"FF3 zip not found: {zip_path}")

    lines = _read_first_csv_from_zip(zip_path)
    df = _parse_monthly_table(lines)

    # Defensive cleanup: strip column names (French files sometimes have spaces)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"Mkt-RF", "SMB", "HML", "RF"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in FF3 table: {missing}. Got: {list(df.columns)}")

    factors = df[["Mkt-RF", "SMB", "HML"]].copy()
    rf = df["RF"].copy()
    rf.name = "RF"

    # Basic sanity checks (optional but helpful)
    if factors.isna().any().any() or rf.isna().any():
        # Not fatal; but usually indicates parsing boundary issues
        pass

    return factors, rf