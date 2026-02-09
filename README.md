# ML_in_Finance_ENSAE
Deep Learning in Asset Pricing

Note importante sur l'arborescence : 
Les fichiers .py vont dans src/ml_in_finance_ensae/
Les notebooks .ipynb restent en dehors de src/

Guide à l'installation :

A la racine du repo (important) faire dans un Terminal "pip install -e ." (le "." à la fin est indispensable) : soit chaque fois que vous utilisez un environnement nouveau (typiquement le cas sur SSPCLOUD) soit une fois pour toute si vous codez toujours sur le même environnement. Pour expliquer : cela permet à l'environnement de faire comprendre qu'on utilise la partie "src" comme un package interne, de sorte que tous les imports se fassent correctement.

Pour importer du .py dans un notebook on fait comme suit (exemple, à adapter en fonction des imports souhaités) : 
from ml_in_finance_ensae.data import load_data
from ml_in_finance_ensae.models import fit_linear_model
