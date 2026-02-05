# src/ml_in_finance_ensae/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import re
import zipfile
import requests
import pandas as pd


@dataclass(frozen=True)
class FrenchPaths:
    root: Path = Path("data")
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")


# URLs "ftp" les plus standards pour ces datasets
FF25_ZIP_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_5x5_CSV.zip"
FF3_ZIP_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def _download(url: str, dest: Path, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest  # cache local
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _read_french_csv_from_zip(zip_path: Path) -> str:
    """Retourne le contenu texte du premier fichier .csv dans le zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"Aucun CSV trouvé dans {zip_path.name}. Contenu: {zf.namelist()}")
        # généralement le 1er est celui qu'on veut
        with zf.open(csv_names[0]) as f:
            return f.read().decode("latin1", errors="replace")


def _parse_french_monthly_table(text: str) -> pd.DataFrame:
    """
    Parse une table mensuelle Fama-French typique (lignes YYYYMM puis colonnes numériques),
    en s'arrêtant quand les lignes ne ressemblent plus à des observations.
    """
    lines = text.splitlines()

    # 1) on trouve la première ligne qui commence par "Date" ou un YYYYMM
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Date"):
            start_idx = i
            break
        if re.match(r"^\s*\d{6}\s*,", line):
            start_idx = i - 1  # on suppose que la ligne header est juste avant
            break
    if start_idx is None:
        raise ValueError("Impossible de trouver le début de table (Date / YYYYMM).")

    # 2) on reconstruit un pseudo-csv à partir de là et on lit
    chunk = "\n".join(lines[start_idx:])

    df = pd.read_csv(io.StringIO(chunk))

    # 3) on garde uniquement les lignes où la première colonne est YYYYMM
    date_col = df.columns[0]
    mask = df[date_col].astype(str).str.match(r"^\d{6}$")
    df = df.loc[mask].copy()

    # 4) date au format Period (mensuel)
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m")
    df = df.set_index(date_col).sort_index()

    # 5) valeurs: souvent en % -> on convertit en décimal
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df / 100.0

    df = df[~df.index.duplicated(keep="first")].sort_index()

    return df


def load_ff25_and_rf(paths: FrenchPaths = FrenchPaths()) -> tuple[pd.DataFrame, pd.Series]:
    """
    Retourne:
    - ff25_excess: DataFrame (T x 25) d'excess returns des 25 portefeuilles
    - rf: Series (T,) risk-free mensuel (en décimal)
    """
    # Téléchargement
    ff25_zip = _download(FF25_ZIP_URL, paths.raw / "25_Portfolios_5x5_CSV.zip")
    ff3_zip = _download(FF3_ZIP_URL, paths.raw / "F-F_Research_Data_Factors_CSV.zip")

    # Lecture + parsing
    ff25_text = _read_french_csv_from_zip(ff25_zip)
    ff3_text = _read_french_csv_from_zip(ff3_zip)

    ff25 = _parse_french_monthly_table(ff25_text)
    ff3 = _parse_french_monthly_table(ff3_text)

    # RF est une colonne du fichier FF3
    if "RF" not in ff3.columns:
        raise ValueError(f"Colonne RF introuvable dans FF3. Colonnes: {list(ff3.columns)}")

    rf = ff3["RF"].rename("RF")

    rf = rf[~rf.index.duplicated(keep="first")].sort_index()
    ff25 = ff25[~ff25.index.duplicated(keep="first")].sort_index()

    # Alignement des dates
    common_idx = ff25.index.intersection(rf.index)
    ff25 = ff25.loc[common_idx]
    rf = rf.loc[common_idx]

    # Excess returns
    ff25_excess = ff25.sub(rf, axis=0)

    return ff25_excess, rf
