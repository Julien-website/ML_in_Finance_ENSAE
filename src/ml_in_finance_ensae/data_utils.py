import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class AssetPricingDataset(Dataset):
    def __init__(self, df_ret, df_macro, df_char, lookback=12):
        """
        Args:
            df_ret: DataFrame des rendements (T, N)
            df_macro: DataFrame des variables macro (T, M)
            df_char: DataFrame ou Array des caractéristiques (T, N, K) 
                     où K est le nombre de caractéristiques (Size, BM, etc.)
            lookback: Fenêtre temporelle pour le LSTM
        """
        super().__init__()
        self.lookback = lookback
    
        # Rendements (Cibles)
        self._y_targets = df_ret.values.astype(np.float32)
        
        # Séries Macro (Input LSTM)
        self._x_macro = df_macro.values.astype(np.float32)
        
        if isinstance(df_char, pd.DataFrame):
            self._x_characteristics = df_char.values.astype(np.float32)
        else:
            self._x_characteristics = df_char.astype(np.float32)
        
        self.n_assets = self._y_targets.shape[1]
        self._len = int(self._y_targets.shape[0])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        x_seq = self._x_macro[idx : idx + self.lookback]
        
        # 2. Caractéristiques des actifs à l'instant t-1 (ou t selon ton décalage)
        # On prend les caractéristiques qui conditionnent le poids du SDF
        x_char = self._x_characteristics[idx] 
        
        # 3. Le rendement correspondant à l'instant t
        y_val = self._y_targets[idx]
        
        return (torch.from_numpy(x_char), 
                torch.from_numpy(x_seq), 
                torch.from_numpy(y_val))

def apply_stationarity_transforms(data: pd.DataFrame, tcodes: pd.Series) -> pd.DataFrame:
    """
    Applique les transformations stationnaires selon les codes FRED-MD.
    
    Args:
        data: DataFrame brut (T x M)
        tcodes: Series indexée par colonnes contenant les codes (float ou int)
        
    Returns:
        DataFrame transformé et nettoyé des NaN initiaux.
    """
    df_transformed = pd.DataFrame(index=data.index)
    
    # Définition des opérations selon le tcode
    # x représente la série temporelle (pd.Series)
    transforms = {
        1: lambda x: x,                                         # No transform
        2: lambda x: x.diff(),                                  # Delta x
        3: lambda x: x.diff().diff(),                           # Delta^2 x
        4: lambda x: np.log(x),                                 # log(x)
        5: lambda x: np.log(x).diff(),                          # Delta log(x)
        6: lambda x: np.log(x).diff().diff(),                   # Delta^2 log(x)
        7: lambda x: (x / x.shift(1) - 1.0).diff()              # Delta (x_t/x_{t-1} - 1)
    }

    for col in data.columns:
        if col in tcodes:
            code = int(tcodes[col])  # Conversion du float en int
            if code in transforms:
                df_transformed[col] = transforms[code](data[col])
            else:
                df_transformed[col] = data[col] # Par défaut, pas de transformation
        else:
            df_transformed[col] = data[col]

    # Supprimer les premières lignes qui contiennent des NaN à cause des diff()
    return df_transformed.dropna()



def add_group_stats(df_macro):
    """
    Calcule des statistiques transversales par groupe de variables.
    df_macro: DataFrame (T x M) déjà stationnarisé.
    """
    # Liste simplifiée des groupes FRED-MD (à adapter selon tes colonnes)
    # Dans le fichier FRED-MD, les colonnes sont souvent ordonnées par groupe.
    # Voici une version robuste basée sur les thématiques :
    groups = {
        'output': ['INDPRO', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD', 'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S', 'IPFUELS'],
        'labor': ['PAYEMS', 'USPRIV', 'MANEMP', 'SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 'USFIRE', 'USGOVT', 'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP514', 'UEMP1526', 'UEMP27OV'],
        'housing': ['HOUST', 'HOUSTNE', 'HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE', 'PERMITMW', 'PERMITS', 'PERMITW'],
        'money': ['M1SL', 'M2SL', 'M2REAL', 'AMBSL', 'TOTRESNS', 'NONBORRES', 'BUSLOANS', 'REALLN', 'NONREVSL', 'CONSREALLN'],
        'interest': ['FEDFUNDS', 'CP3MX', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 'COMPAPFF', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 'BAAFFM'],
        'prices': ['PPIFGS', 'PPIITM', 'PPIBEG', 'CPIAUCSL', 'CPIAPPSL', 'CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD', 'CUSR0000SAS', 'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5', 'PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA']
    }

    df_final = df_macro.copy()

    for name, cols in groups.items():
        # On ne garde que les colonnes qui existent réellement dans ton fichier
        existing_cols = [c for c in cols if c in df_macro.columns]
        
        if len(existing_cols) > 0:
            # Médiane du groupe : signal central du secteur
            df_final[f'grp_{name}_median'] = df_macro[existing_cols].median(axis=1)
            # Volatilité du groupe : incertitude du secteur
            df_final[f'grp_{name}_std'] = df_macro[existing_cols].std(axis=1)

    return df_final