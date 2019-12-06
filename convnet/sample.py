from environment import Environment

sampleNumber = 8000
DIM = 10
SHIPS = [2,3,3,4,5]
e = Environment(DIM, SHIPS, "Vikram")

e.save_sample(sampleNumber)