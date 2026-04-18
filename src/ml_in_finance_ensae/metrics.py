import numpy as np
import pandas as pd
import torch

def calculate_sharpe_ratio(portfolio_returns, annualization_factor=np.sqrt(12)):
    """
    Calcule le ratio de Sharpe du portefeuille SDF (le facteur stochastique).
    """
    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns)
    if sigma == 0: return 0
    return (mu / sigma) * annualization_factor

def calculate_r2_scores(realized_returns, predicted_returns):
    """
    realized_returns: np.array 1D (all stocks * all months)
    predicted_returns: np.array 1D (all stocks * all months)
    """
    # 1. TOTAL R2 (Global)
    # Dans l'article, on compare à la prédiction 0 (pas de rendement moyen)
    numerator = np.sum((realized_returns - predicted_returns)**2)
    denominator = np.sum(realized_returns**2)
    
    r2_total = 1 - (numerator / denominator)

    # 2. CROSS-SECTIONAL R2 (XS)
    # L'article l'évalue souvent sur les rendements moyens par actif
    # Problème : avec np.concatenate, on a perdu l'index des actifs.
    # Pour un vrai XS R2, il faudrait la corrélation ou l'erreur sur les moyennes.
    
    # Approche simplifiée (Corrélation de Spearman / Classement)
    # L'article valorise la capacité à ordonner les actifs.
    from scipy.stats import spearmanr
    corr_xs, _ = spearmanr(realized_returns, predicted_returns)
    
    return r2_total, corr_xs


def evaluate_performance(trainer, dataloader, device):
    trainer.sdf_net.eval()
    trainer.lstm_net.eval()
    
    all_omega = []
    all_returns = []
    sdf_returns = []

    with torch.no_grad():
        for char, macro, ret in dataloader:
            char, macro, ret = char.to(device), macro.to(device), ret.to(device)
            
            # Nettoyage des dimensions [1, N, C] -> [N, C]
            char = char.squeeze(0)
            ret = ret.squeeze(0)
            
            # 1. Forward pass
            h_t = trainer.lstm_net(macro)
            omega = trainer.sdf_net(char, h_t) # Shape [N, 1]
            
            # 2. Calcul du rendement du SDF pour ce mois t
            # On s'assure que ret et omega ont la même forme pour la multiplication
            r_sdf_t = torch.sum(omega.view(-1) * ret.view(-1)).item()
            
            # 3. APPEND (Le remplissage des listes)
            sdf_returns.append(r_sdf_t)
            all_omega.append(omega.cpu().numpy().flatten())
            all_returns.append(ret.cpu().numpy().flatten())

    # --- CALCULS APRÈS LA BOUCLE ---
    
    # Conversion et concaténation (pour gérer les N qui varient chaque mois)
    all_omega_flat = np.concatenate(all_omega)
    all_returns_flat = np.concatenate(all_returns)
    sdf_returns = np.array(sdf_returns)

    # Nettoyage final contre les NaNs résiduels dans les données
    mask = np.isfinite(sdf_returns)
    sdf_returns = sdf_returns[mask]

    if len(sdf_returns) < 2:
        return {"Sharpe_Ratio": 0, "R2_Total": 0}

    # Calcul du Sharpe
    mu = np.mean(sdf_returns)
    sigma = np.std(sdf_returns)
    sharpe = (mu / sigma) * np.sqrt(12) if sigma > 1e-9 else 0
    
    # Calcul des R2 (avec vos fonctions)
    r2_total, r2_xs = calculate_r2_scores(all_returns_flat, all_omega_flat)
    
    return {
        "Sharpe_Ratio": sharpe,
        "R2_Total": r2_total,
        "R2_CrossSectional": r2_xs,
        "SDF_Volatility": sigma * np.sqrt(12)
    }


def plot_characteristic_importance(trainer, char_data, macro_history, char_names):
    """
    Calcule l'importance par sensibilité (gradients ou magnitude des poids).
    """
    trainer.sdf_net.eval()
    h_t = trainer.lstm_net(macro_history)
    
    # On regarde la magnitude moyenne des poids omega par caractéristique
    # Dans un modèle linéaire, omega = beta. Ici c'est une approximation locale.
    with torch.no_grad():
        weights = trainer.sdf_net(char_data, h_t)
        
    # Analyse de sensibilité simple : importance = moyenne absolue des poids
    importance = np.abs(weights.cpu().numpy()).mean(axis=0)
    
    # Création du graphique
    feat_imp = pd.Series(importance, index=char_names).sort_values(ascending=False)
    feat_imp.head(10).plot(kind='barh', title="Top 10 Characteristics Importance")


def evaluate_performance_ensemble(trainer, dataloader, device):
    """
    Évalue la performance en utilisant la moyenne de l'ensemble.
    """
    all_omega = []
    all_returns = []
    sdf_returns_list = []

    with torch.no_grad():
        for char, macro, ret in dataloader:
            char, macro, ret = char.to(device), macro.to(device), ret.to(device)
            
            # 1. Prédiction par consensus
            omega = trainer.predict_ensemble_weights(char, macro) # (N, 1)
            
            # 2. Rendement du SDF
            ret_flat = ret.squeeze(0)
            r_sdf_t = torch.sum(omega * ret_flat).item()
            
            sdf_returns_list.append(r_sdf_t)
            all_omega.append(omega.cpu().numpy())
            all_returns.append(ret_flat.cpu().numpy())

    # Calcul du Sharpe Global
    sdf_returns_list = np.array(sdf_returns_list)
    mu = np.mean(sdf_returns_list)
    sigma = np.std(sdf_returns_list)
    sharpe = (mu / sigma) * np.sqrt(12) if sigma > 1e-9 else 0
    
    return {
        "Sharpe_Ratio": sharpe,
        "all_omega": all_omega,
        "all_returns": all_returns,
        "sdf_returns_list": sdf_returns_list
    }