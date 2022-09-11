import numpy as np
import pandas as pd
import sys
import util
import matplotlib.pyplot as plt

# Check the command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# Learning rate
t = 0.001

# Limit interations
max_steps = 1000

# Get the arg and read in the spreadsheet
infilename = sys.argv[1]
X, Y, labels = util.read_excel_data(infilename)
n, d = X.shape
print(f"Read {n} rows, {d - 1} features from '{infilename}'.")

# Get the mean and standard deviation for each column
## Your code here
x_mean = np.mean(X, axis=0)
x_std= np.std(X, axis=0)

# Don't mess with the first column (the 1s)
## Your code here
x_mean[0] = 0
x_std[0] = 1
# Standardize X to be X' 
Xp = np.subtract(X, x_mean) 
Xp = np.divide(Xp, x_std) 
# function MSE
def Mean_Square_Error(y_true,y_pred):
    MSE = np.sum((y_true-y_pred)**2) / len(y_true)
    return MSE
# First guess for B is "all coefficents are zero"
B_standard = np.zeros(6)

# Create a numpy array to record avg error for each step
errors = np.zeros(max_steps)
R2_coverage = 0.874489
current_mse = 0
for i in range(max_steps):
    learning_rate = -0.001
    
    # Compute the gradient 
    ## Your code here
    y_predict = np.dot(Xp,B_standard)
    gradient = -(2/n) *np.sum(X.T*(Y - y_predict))
    # Compute a new B (use `t`)
    B_standard = B_standard - (learning_rate * gradient)    
    # Figure out the average squared error using the new B
    y_predict_2 = np.dot(Xp,B_standard)
    previous_mse = current_mse
    current_mse = Mean_Square_Error(Y, y_predict_2)
    # Store it in `errors``
    errors[i] = current_mse
    # Check to see if we have converged
    if previous_mse and abs(previous_mse-current_mse)<=0.001:
        break

print(f"Took {i} iterations to converge")

# "Unstandardize" the coefficients
## Your code here
B = np.divide(B_standard, x_std)
# Show the result
print(util.format_prediction(B, labels))

# Get the R2 score
R2 = util.score(B, X, Y)
print(f"R2 = {R2:f}")

# Draw a graph
fig1 = plt.figure(1, (4.5, 4.5))
## Your code ehre

fig1.plot(weights, costs, marker='o', color='red')
fig1.title("Cost vs Weights")
fig1.ylabel("Cost")
fig1.xlabel("Weight")
fig1.show()
fig1.savefig("err.png")
