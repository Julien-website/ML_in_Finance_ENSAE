import torch
import torch.nn as nn
import numpy as np

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
        """
        Une étape d'entraînement (pour un mois t donné).
        """
        # --- 1. MISE À JOUR DE L'ADVERSAIRE (Maximisation de l'erreur) ---
        self.opt_adv.zero_grad()
        
        # On ne veut pas calculer les gradients pour le SDF ici
        with torch.no_grad():
            h_t = self.lstm_net(macro_history)
            weights = self.sdf_net(char_data, h_t)
        
        # On calcule les instruments
        h_t_adv = self.lstm_net(macro_history).detach() # On détache le LSTM
        instruments = self.adv_net(char_data, h_t_adv)
        
        # L'adversaire veut MAXIMISER l'erreur, donc on MINIMISE l'opposé
        loss_adv = -self.pricing_error_loss(weights, returns, instruments)
        loss_adv.backward()
        self.opt_adv.step()

        # --- 2. MISE À JOUR DU SDF (Minimisation de l'erreur) ---
        self.opt_sdf.zero_grad()
        
        h_t = self.lstm_net(macro_history)
        weights = self.sdf_net(char_data, h_t)
        
        # On ne veut pas calculer les gradients pour l'Adversaire ici
        with torch.no_grad():
            instruments = self.adv_net(char_data, h_t)
            
        loss_sdf = self.pricing_error_loss(weights, returns, instruments)
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