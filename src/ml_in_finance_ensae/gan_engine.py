import torch
import torch.nn as nn
import numpy as np
import copy


class GANTrainer:
    def __init__(self, sdf_net, adv_net, lstm_net, lr_sdf=1e-4, lr_adv=1e-3):
        """
        Gère l'entraînement antagoniste.
        """
        self.sdf_net = sdf_net
        self.adv_net = adv_net
        self.lstm_net = lstm_net
        
        # Optimiseurs : Le SDF et le LSTM sont entraînés ensemble
        self.opt_sdf = torch.optim.Adam(
            list(self.sdf_net.parameters()) + list(self.lstm_net.parameters()), 
            lr=lr_sdf
        )
        # L'adversaire a son propre optimiseur (souvent un LR plus élevé)
        self.opt_adv = torch.optim.Adam(self.adv_net.parameters(), lr=lr_adv)

    def pricing_error_loss(self, weights, returns, instruments):
        """
        Calcule la norme au carré de l'erreur de pricing : ||E[M * R * g]||^2
        weights : omega (N, 1)
        returns : R_e (N, 1)
        instruments : g (N, K)
        """
        # 1. Calcul du SDF : M = 1 - sum(omega_i * R_i)
        # Note : On calcule un seul M pour tout le batch (le mois t)
        sdf = 1.0 - torch.sum(weights * returns)
        
        # 2. Erreur de moment pour chaque instrument k : E[M * R * g_k]
        # returns * instruments -> (N, K)
        # sdf * (returns * instruments) -> (N, K)
        moments = torch.mean(sdf * returns * instruments, dim=0) # Moyenne sur les N actifs
        
        # 3. Loss = Somme des carrés des moments
        return torch.sum(moments**2)

    def train_step(self, char_data, macro_history, returns):
        char_data = char_data.squeeze(0) 
        returns = returns.squeeze(0)
        
        # On ne squeeze PAS macro_history car le LSTM veut ses 3 dimensions [1, 12, 138]
        # ----------------------------------

        # --- 1. MISE À JOUR DE L'ADVERSAIRE ---
        self.opt_adv.zero_grad()
        
        h_t_fixed = self.lstm_net(macro_history).detach()
        # Maintenant char_data est en 2D, donc l'erreur disparaît !
        weights_fixed = self.sdf_net(char_data, h_t_fixed).detach()
        
        # L'adversaire est actif
        instruments = self.adv_net(char_data, h_t_fixed)
        
        loss_adv = -self.pricing_error_loss(weights_fixed, returns, instruments)
        loss_adv.backward()
        self.opt_adv.step()

        # --- 2. MISE À JOUR DU SDF ET DU LSTM ---
        self.opt_sdf.zero_grad()
        
        # Le LSTM et le SDF sont actifs (pas de detach ici !)
        h_t = self.lstm_net(macro_history)
        weights = self.sdf_net(char_data, h_t)
        
        # L'adversaire est figé
        with torch.no_grad():
            # On utilise le h_t tout juste calculé mais on détache l'adv_net
            instruments_fixed = self.adv_net(char_data, h_t)
            
        loss_sdf = self.pricing_error_loss(weights, returns, instruments_fixed)
        loss_sdf.backward()
        self.opt_sdf.step()

        return loss_sdf.item(), -loss_adv.item()
    

    def predict_expected_return(self, char_data, macro_history):
        """
        Calcule l'espérance de rendement conditionnelle mu_it
        """
        self.sdf_net.eval()
        self.lstm_net.eval()
        with torch.no_grad():
            h_t = self.lstm_net(macro_history)
            weights = self.sdf_net(char_data, h_t)
            # En théorie : E[R] = -Cov(R, M) / E[M]
            # Ici on peut approximer la prime de risque par les poids omega
            # ou par simulation si on a accès à plusieurs scénarios.
        return weights # Simplification : les poids omega sont proportionnels à E[R]
    

class GANTrainerEnsemble:
    def __init__(self, sdf_net, adv_net, lstm_net, lr_sdf=1e-4, lr_adv=1e-3, n_models=5):
        """
        Gère un ensemble de modèles GAN pour stabiliser l'apprentissage.
        """
        self.n_models = n_models
        self.device = next(sdf_net.parameters()).device
        self.models = []

        for i in range(n_models):
            # On clone l'architecture de base pour chaque membre de l'ensemble
            m = {
                'sdf': copy.deepcopy(sdf_net).to(self.device),
                'adv': copy.deepcopy(adv_net).to(self.device),
                'lstm': copy.deepcopy(lstm_net).to(self.device),
            }
            # Optimiseurs individuels pour chaque membre
            m['opt_sdf'] = torch.optim.Adam(
                list(m['sdf'].parameters()) + list(m['lstm'].parameters()), 
                lr=lr_sdf, weight_decay=1e-4 # L2 pour la stabilité
            )
            m['opt_adv'] = torch.optim.Adam(m['adv'].parameters(), lr=lr_adv)
            self.models.append(m)

    def pricing_error_loss(self, weights, returns, instruments):
        # 1. Calcul du SDF M = 1 - sum(w * R)
        # weights: (N, 1), returns: (N, 1) -> scalaire
        sdf = 1.0 - torch.sum(weights * returns)
        
        # Protection numérique
        sdf = torch.clamp(sdf, min=-2, max=2) 
        
        # 2. Calcul du rendement actualisé pour chaque actif (N, 1)
        # On s'assure que 'returns' est bien une colonne pour le broadcasting
        pricing_impact = sdf * returns.view(-1, 1) 
        
        # 3. Alignement pour les instruments (N, 8)
        # pricing_impact (N, 1) est "étendu" virtuellement en (N, 8) 
        # pour multiplier chaque instrument de chaque actif
        moments_per_asset = pricing_impact.expand_as(instruments) * instruments
        
        # 4. Condition de moment : moyenne sur les N actifs -> vecteur de taille 8
        moments = torch.mean(moments_per_asset, dim=0)
        
        # La perte est la somme des carrés des 8 erreurs de prix
        return torch.sum(moments**2)

    def train_step(self, char_data, macro_history, returns):
        char_data = char_data.squeeze(0)
        returns = returns.squeeze(0)
        
        total_loss_sdf = 0
        total_loss_adv = 0
        
        # On entraîne chaque modèle de l'ensemble séparément
        for m in self.models:
            # --- 1. UPDATE ADVERSAIRE ---
            m['opt_adv'].zero_grad()
            with torch.no_grad():
                h_t_fixed = m['lstm'](macro_history).detach()
                w_fixed = m['sdf'](char_data, h_t_fixed).detach()
            
            inst = m['adv'](char_data, h_t_fixed)
            l_adv = -self.pricing_error_loss(w_fixed, returns, inst)
            l_adv.backward()
            torch.nn.utils.clip_grad_norm_(m['adv'].parameters(), 1.0)
            m['opt_adv'].step()

            # --- 2. UPDATE SDF + LSTM ---
            m['opt_sdf'].zero_grad()
            h_t = m['lstm'](macro_history)
            w = m['sdf'](char_data, h_t)
            with torch.no_grad():
                inst_fixed = m['adv'](char_data, h_t).detach()
            
            l_sdf = self.pricing_error_loss(w, returns, inst_fixed)
            l_sdf.backward()
            torch.nn.utils.clip_grad_norm_(m['sdf'].parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(m['lstm'].parameters(), 1.0)
            m['opt_sdf'].step()
            
            total_loss_sdf += l_sdf.item()
            total_loss_adv += -l_adv.item()

        return total_loss_sdf / self.n_models, total_loss_adv / self.n_models

    def predict_ensemble_weights(self, char_data, macro_history):
        """
        Calcule la moyenne des poids omega sur tout l'ensemble.
        """
        all_weights = []
        for m in self.models:
            m['sdf'].eval()
            m['lstm'].eval()
            with torch.no_grad():
                h_t = m['lstm'](macro_history)
                w = m['sdf'](char_data.squeeze(0), h_t)
                all_weights.append(w)
        
        # Retourne le consensus (moyenne des poids)
        return torch.stack(all_weights).mean(dim=0)