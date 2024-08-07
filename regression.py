import csv
import numpy as np
import matplotlib.pyplot as plt

# Lecture du fichier CSV
def lire_csv(fichier):
	#check if file exists
	try:
		open(fichier, 'r')
	except FileNotFoundError:
		print("File not found")
		return np.array([]), np.array([])
	with open(fichier, 'r') as fichier_csv:
		reader = csv.DictReader(fichier_csv)
		prix = []
		kilometre = []
		for ligne in reader:
			prix.append(float(ligne['price']))
			kilometre.append(float(ligne['km']))
	return np.array(prix), np.array(kilometre)

# Régression linéaire "from scratch"
def linear_regression(x, y):
	n = len(x)
	x_moy = np.mean(x)
	y_moy = np.mean(y)
	num = np.sum((x - x_moy) * (y - y_moy))
	den = np.sum((x - x_moy) ** 2)
	b1 = num / den
	b0 = y_moy - b1 * x_moy
	return b0, b1

def gradient_descent(x, y, learningRate, nbIterations):
	m = len(x)
	b0 = 0
	b1 = 0
	
	for i in range(nbIterations):
		y_pred = b0 + b1 * x
		erreur = y_pred - y
		gradient_b0 = (1/m) * np.sum(erreur)
		gradient_b1 = (1/m) * np.sum(erreur * x)
		
		b0 = b0 - learningRate * gradient_b0
		b1 = b1 - learningRate * gradient_b1
	
	return b0, b1

def calculate_accuracy(x, y, b0, b1):
    y_pred = b0 + b1 * x
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    mae = np.mean(np.abs(y - y_pred))
    
    return mse, rmse, r2, mae

# Visualisation
def visualiser(x, y, b0, b1):
	plt.scatter(x, y, label='Données')
	x_pred = np.linspace(x.min(), x.max(), 100)
	y_pred = b0 + b1 * x_pred
	plt.plot(x_pred, y_pred, label='Régression linéaire', color='red')
	plt.xlabel('Kilomètre')
	plt.ylabel('Prix')
	plt.title('Régression linéaire')
	plt.legend()
	plt.show()

# Programme principal
def main():
	fichier_csv = 'data.csv'
	# fichier_csv ='data_from_variables.csv'
	# fichier_csv = input("Enter the filename: ")
	price, kilometer = lire_csv(fichier_csv)
	if len(price) == 0 or len(kilometer) == 0:
		print("Empty file")
		return
	
	price_normalized = (price - np.min(price)) / (np.max(price) - np.min(price))
	kilometer_normalized = (kilometer - np.min(kilometer)) / (np.max(kilometer) - np.min(kilometer))
	# b0, b1 = regression_lineaire(kilometre, prix)
	# print(f'Équation de la droite : y = {b0:.2f} + {b1:.2f}x')
	# visualiser(kilometre, prix, b0, b1)
	b0, b1 = gradient_descent(kilometer_normalized, price_normalized, 0.1, 10000)
	b0_denorm = b0 * (np.max(price) - np.min(price)) + np.min(price)
	b1_denorm = b1 * (np.max(price) - np.min(price)) / (np.max(kilometer) - np.min(kilometer))
	b0,b1 = b0_denorm , b1_denorm
	
	print(f'y = {b0:.2f} + {b1:.2f}x')
	mse, rmse, r2, mae = calculate_accuracy(kilometer, price, b0, b1)
	print(f'MSE: {mse:.2f}')
	print(f'RMSE: {rmse:.2f}')
	print(f'R2: {r2:.2f}')
	print(f'MAE: {mae:.2f}')

	#put b0 and b1 in a csv file
	with open('b0_b1.csv', 'w') as fichier_csv:
		writer = csv.writer(fichier_csv)
		writer.writerow(['b0', 'b1'])
		writer.writerow ([b0, b1])
	visualiser(kilometer, price, b0, b1)

# mse : très proche de 0, signifie que les prédictions sont très précises et que l'erreur est minimale.
# rmse : très proche de 0, signifie que la racine carrée de l'erreur moyenne est également très proche de 0.
# r2 : très proche de 1, signifie que le modèle explique presque toute la variance des données.
# mae : très proche de 0, signifie que l'erreur absolue moyenne est également très proche de 0.
if __name__ == '__main__':
	main()