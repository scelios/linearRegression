import csv
import numpy as np
import matplotlib.pyplot as plt

# Lecture du fichier CSV
def lire_csv(fichier):
	with open(fichier, 'r') as fichier_csv:
		reader = csv.DictReader(fichier_csv)
		b0 = 0
		b1 = 0
		for ligne in reader:
			b0 = float(ligne['b0'])
			b1 = float(ligne['b1'])
	return b0, b1

# Programme principal
def main():
	# fichier_csv = 'data.csv'
	fichier_csv ='b0_b1.csv'
	# enter number of kilometer
	b0, b1 = lire_csv(fichier_csv)
	print(f'y = {b0:.2f} + {b1:.2f}x')
	number = input("Enter number of kilometer: ")
	print(f'Price for {number} km: {b0 + b1 * float(number):.2f}')

if __name__ == '__main__':
	main()