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
            #layers.append(nn.BatchNorm1d(h_dim)) # Optionnel mais aide à la stabilité
            layers.append(nn.ReLU())
            # L'article mentionne un dropout très élevé (conservation de 95%, soit p=0.05)
            layers.append(nn.Dropout(p=0.05)) 
            in_dim = h_dim
            
        # La dernière couche produit un scalaire : le poids omega_i,t
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, char_data, macro_states):
        """
        char_data: [N, char_dim]
        macro_states: [1, macro_dim] venant du LSTM
        """
        # Force macro_states à être [1, macro_dim] au cas où il y aurait des dimensions parasites
        if macro_states.dim() > 2:
            macro_states = macro_states.view(-1, macro_states.size(-1))
        
        # On duplique l'état macro pour chaque ligne (action) de char_data
        # macro_repeated devient [N, macro_dim]
        n_stocks = char_data.size(0)
        macro_repeated = macro_states.expand(n_stocks, -1) 

        # Concaténation : [N, char_dim + macro_dim]
        combined_input = torch.cat([char_data, macro_repeated], dim=1)
        
        return self.output_layer(self.network(combined_input))

        
class AdversarialNetwork(nn.Module):
    def __init__(self, char_dim, macro_dim, output_dim=8):
        super(AdversarialNetwork, self).__init__()
        input_dim = char_dim + macro_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.05), # Dropout de 5% (Garde 95% des neurones)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh()
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
    def __init__(self, input_dim, lstm_hidden_dim):
        super(MacroLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        self.ln = nn.LayerNorm(lstm_hidden_dim) 

    def forward(self, x):
        # x: [1, 12, 138] (Batch, Seq, Features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # h_n est [num_layers, batch, hidden_dim]
        last_step = h_n[-1] # Devient [1, hidden_dim]
        out = self.ln(last_step)
        # On passe par la LayerNorm
        return out