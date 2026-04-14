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
    Calcule deux types de R² mentionnés dans l'article :
    1. Total R² : Mesure la variance totale expliquée.
    2. Predictive R² (Cross-sectional) : Mesure la capacité à classer les actifs.
    """
    # R² Total (immédiat)
    res_total = np.sum((realized_returns - predicted_returns)**2)
    tot_total = np.sum(realized_returns**2) # Comparaison au modèle zéro (sans risque)
    r2_total = 1 - (res_total / tot_total)
    
    # R² Cross-sectional (moyenne par actif sur la période)
    # On agrège par actif (axis 0 dans la matrice T x N)
    mu_realized = np.mean(realized_returns, axis=0)
    mu_predicted = np.mean(predicted_returns, axis=0)
    
    res_xs = np.sum((mu_realized - mu_predicted)**2)
    tot_xs = np.sum((mu_realized - np.mean(mu_realized))**2)
    r2_xs = 1 - (res_xs / tot_xs)
    
    return r2_total, r2_xs

def evaluate_performance(trainer, dataloader, device):
    """
    Fonction maîtresse qui parcourt le dataset et calcule toutes les métriques.
    """
    trainer.sdf_net.eval()
    trainer.lstm_net.eval()
    
    all_omega = []
    all_returns = []
    sdf_returns = []

    with torch.no_grad():
        for char, macro, ret in dataloader:
            char, macro, ret = char.to(device), macro.to(device), ret.to(device)
            
            # 1. Extraction des poids du SDF (omega)
            h_t = trainer.lstm_net(macro)
            omega = trainer.sdf_net(char, h_t) # (N, 1)
            
            # 2. Rendement du portefeuille SDF (poids * rendements réalisés)
            # R_sdf = sum(omega_i * R_i)
            r_sdf_t = torch.sum(omega * ret).item()
            sdf_returns.append(r_sdf_t)
            
            # Stockage pour R² (on repasse en CPU/Numpy)
            all_omega.append(omega.cpu().numpy().flatten())
            all_returns.append(ret.cpu().numpy().flatten())

    # Conversion en matrices (T, N)
    all_omega = np.array(all_omega)
    all_returns = np.array(all_returns)
    
    # Calcul des métriques
    sharpe = calculate_sharpe_ratio(sdf_returns)
    r2_total, r2_xs = calculate_r2_scores(all_returns, all_omega) # omega est ici le proxy de E[R]
    
    metrics = {
        "Sharpe_Ratio": sharpe,
        "R2_Total": r2_total,
        "R2_CrossSectional": r2_xs,
        "SDF_Volatility": np.std(sdf_returns) * np.sqrt(12)
    }
    
    return metrics


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