#librairies
import torch
import torch.nn as nn

class SDFNetwork(nn.Module):
    def __init__(self, char_dim, macro_dim, hidden_units=[64, 32]):
        """
        Args:
            char_dim (int): Nombre de caractéristiques par action (ex: 94 dans le papier).
            macro_dim (int): Nombre d'états macro extraits par le LSTM (ex: 4).
            hidden_units (list): Architecture des couches cachées.
        """
        super(SDFNetwork, self).__init__()
        
        # L'entrée combine les caractéristiques de l'action et l'état de l'économie
        input_dim = char_dim + macro_dim
        
        # Construction des couches
        layers = []
        in_dim = input_dim
        for h_dim in hidden_units:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim)) # Optionnel mais aide à la stabilité
            layers.append(nn.ReLU())
            # L'article mentionne un dropout très élevé (conservation de 95%, soit p=0.05)
            layers.append(nn.Dropout(p=0.05)) 
            in_dim = h_dim
            
        # La dernière couche produit un scalaire : le poids omega_i,t
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, char_data, macro_states):
        """
        Args:
            char_data: Tensor (N, char_dim) - caractéristiques des N actions
            macro_states: Tensor (1, macro_dim) - état macro commun à l'instant t
        """
        # On répète l'état macro pour chaque action pour pouvoir les concaténer
        n_stocks = char_data.shape[0]
        macro_repeated = macro_states.repeat(n_stocks, 1)
        
        # Concaténation : [Caractéristiques Action | États Macro]
        combined_input = torch.cat([char_data, macro_repeated], dim=1)
        
        # Calcul du poids omega
        omega = self.output_layer(self.network(combined_input))
        return omega
    
class AdversarialNetwork(nn.Module):
    def __init__(self, char_dim, macro_dim, output_dim=8):
        super(AdversarialNetwork, self).__init__()
        input_dim = char_dim + macro_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # Génère 'output_dim' instruments
        )

    def forward(self, char_data, macro_states):
        n_stocks = char_data.shape[0]
        macro_repeated = macro_states.repeat(n_stocks, 1)
        combined_input = torch.cat([char_data, macro_repeated], dim=1)
        
        # Sortie : g(I_{t,i})
        return self.network(combined_input)
    
def compute_pricing_error(weights, returns, instruments):
    """
    weights: omega_i,t (N, 1) issus du SDFNet
    returns: R^e_{i,t+1} (N, 1)
    instruments: g_{i,t} (N, output_dim) issus de l'Adversaire
    """
    # M = 1 - sum(omega * R^e)
    sdf = 1 - torch.sum(weights * returns)
    
    # Condition de moment : E[M * R^e * g]
    # On calcule l'erreur pour chaque instrument
    errors = torch.mean(sdf * returns * instruments, dim=0)
    
    # La perte est la somme des carrés des erreurs de prix
    return torch.sum(errors**2)

class MacroLSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim=4, n_layers=1):
        """
        Args:
            input_dim (int): Nombre de séries macro (ex: 178 dans le papier).
            lstm_hidden_dim (int): Taille de l'état latent (le papier suggère 4).
            n_layers (int): Nombre de couches LSTM (généralement 1 ou 2).
        """
        super(MacroLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=n_layers,
            batch_first=True  # Entrée formatée : (batch, seq_len, features)
        )
        
        # Souvent, on ajoute une petite couche de normalisation à la sortie
        self.bn = nn.BatchNorm1d(lstm_hidden_dim)

    def forward(self, macro_sequence):
        """
        Args:
            macro_sequence: Tensor (batch, T_lookback, input_dim)
                            T_lookback est la fenêtre de temps passé (ex: 12 mois).
        """
        # lstm_out: (batch, seq_len, hidden_dim)
        # h_n: (n_layers, batch, hidden_dim) -> contient le dernier état caché
        lstm_out, (h_n, c_n) = self.lstm(macro_sequence)
        
        # On extrait le dernier état (t) pour caractériser l'économie actuelle
        last_state = h_n[-1] 
        
        # Normalisation pour aider le FFN et l'Adversaire à converger
        out = self.bn(last_state)
        return out