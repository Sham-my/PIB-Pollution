import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Charger les données
data_pollution = pd.read_csv("Gaz.csv", delimiter=';', index_col=0)
data_pib = pd.read_csv("PIB_V2.csv", delimiter=";", index_col=0)

# Supprimer les 37 premières lignes de `data_pollution`
data_pollution = data_pollution.iloc[37:].reset_index()
data_pib = data_pib.head(40)

# Renommer les colonnes pour les années 2019
data_pollution = data_pollution.rename(columns={"2019": "pollution_2019"})
data_pib = data_pib.rename(columns={"2019": "PIB_2019"})

# Supprimer les lettres dans les valeurs de `pollution_2019`
data_pollution['pollution_2019'] = data_pollution['pollution_2019'].str.replace(r'[^\d.]', '', regex=True)

# Convertir les colonnes en données numériques
data_pollution['pollution_2019'] = pd.to_numeric(data_pollution['pollution_2019'], errors='coerce')
data_pib['PIB_2019'] = pd.to_numeric(data_pib['PIB_2019'], errors='coerce')

# Fusionner les deux tables pour la régression linéaire
data = pd.merge(data_pollution[['country', 'pollution_2019']], 
                data_pib[['country', 'PIB_2019']], 
                on='country')


# Éviter les doublons
data = data.groupby('country', as_index=False).mean().reset_index(drop=True)

# Supprimer la valeur inapproprié
data = data.drop(11)

data.head(5)

# Variables pour la régression linéaire
x_rl = data['PIB_2019'].values  # Variable indépendante (PIB)
y_rl = data['pollution_2019'].values   # Variable dépendante (Polution)

#  Régression linéaire
coefficients = np.polyfit(x_rl, y_rl, 1)  # Régression linéaire de degré 1
slope, intercept = coefficients  # Pente et intercept

# Prédiction des valeurs de y
y_pred = np.polyval(coefficients, x_rl)

# Afficher les résultats de la régression
print("=== Régression linéaire ===")
print(f"Pente (slope) : {slope}")
print(f"Ordonnée à l'origine (intercept) : {intercept}")
correlation = np.corrcoef(x_rl, y_rl)[0, 1]
print(f"Coefficient de corrélation : {correlation}")

# Visualisation de la régression linéaire avec les noms des pays
plt.figure(figsize=(10, 6))
plt.scatter(x_rl, y_rl, color='blue', label='Données observées')

# Ajouter une droite de régression
plt.plot(x_rl, y_pred, color='red', label='Régression linéaire')

# Ajouter les noms des pays
for i, country in enumerate(data['country']):
    plt.text(x_rl[i], y_rl[i], country, fontsize=8, ha='right', va='bottom')

# Étiquettes et titre
plt.xlabel('PIB 2019')
plt.ylabel('Pollution 2019')
plt.title('Régression linéaire : Pollution vs PIB')
plt.legend()
plt.grid(alpha=0.5)
plt.show()
