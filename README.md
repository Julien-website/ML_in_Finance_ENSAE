# ML_in_Finance_ENSAE
Deep Learning in Asset Pricing

Note importante sur l'arborescence : 
Les fichiers .py vont dans src/ml_in_finance_ensae/
Les notebooks .ipynb restent en dehors de src/

A la racine du repo faire dans un Terminal "pip install -e"

Pour importer du .py dans un notebook on fait comme suit (exemple, à adapter en fonction des imports souhaités) : 
from ml_in_finance_ensae.data import load_data
from ml_in_finance_ensae.models import fit_linear_model
