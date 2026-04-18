import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class AssetPricingDataset(Dataset):
    def __init__(self, returns, macro, lookback=12):
        """
        Args:
            returns (pd.DataFrame): T x N (Excess returns des 84 portefeuilles)
            macro (pd.DataFrame): T x M (Variables macro stationnaires)
            lookback (int): Fenêtre temporelle pour le LSTM
        """
        # 1. Alignement des dates
        common_dates = returns.index.intersection(macro.index)
        self.returns_df = returns.loc[common_dates]
        self.macro_df = macro.loc[common_dates]
        
        self.lookback = lookback
        # On ne peut commencer qu'après avoir assez de recul pour le LSTM
        self.dates = common_dates[lookback:] 
        
        # 2. Matrice Identité (84 x 84)
        # Puisque ce sont des portefeuilles, leurs "caractéristiques" sont 
        # simplement leur identité (One-hot encoding)
        self.n_assets = len(returns.columns)
        self.identity = torch.eye(self.n_assets)

    def __len__(self):
        # On prédit t+1, donc on s'arrête à l'avant-dernière date disponible
        return len(self.dates) - 1

    def __getitem__(self, idx):
        date_t = self.dates[idx]
        date_t_plus_1 = self.dates[idx + 1]
        
        # 1. Macro (Fenêtre glissante) : t-lookback+1 jusqu'à t
        macro_seq = self.macro_df.loc[:date_t].iloc[-self.lookback:]
        macro_values = macro_seq.values.astype(np.float32)
        macro_tensor = torch.tensor(macro_values, dtype=torch.float32)
        
        # 2. Caractéristiques (SDF Input)
        # On renvoie toujours la même matrice identité car les actifs sont les mêmes
        chars_tensor = self.identity

        # 3. Rendements à t+1 (Target)
        # On récupère les rendements des 84 actifs à la date suivante
        returns_t_plus_1 = self.returns_df.loc[date_t_plus_1]
        returns_values = returns_t_plus_1.values.astype(np.float32)
        returns_tensor = torch.tensor(returns_values, dtype=torch.float32).unsqueeze(1)
        
        return chars_tensor, macro_tensor, returns_tensor

    


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