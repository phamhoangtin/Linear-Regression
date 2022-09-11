from cProfile import label
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
# Read in the excel file
# Returns:
#   X: first column is 1s, the rest are from the spreadsheet
#   Y: The last column from the spreadsheet
#   labels: The list of headers for the columns of X from the spreadsheet
def read_excel_data(infilename):
    ## Your code here
    data_excel = pd.read_excel(f"./{infilename}",sheet_name = 'properties', header = 0,)
    X_df = data_excel.iloc[:,:-1]
    X_df['property_id'] =1
    labels = list(X_df)
    Y_df = data_excel["price"]
    X = X_df.to_numpy()
    Y = Y_df.to_numpy()
    return X, Y, labels


# Make it pretty
def format_prediction(B, labels):
    ## Your code here
    ##predicted price = $32,362.85 + ($85.61 x sqft_hvac) + ($2.73 x sqft_yard) +
    ##($59,195.07 x bedrooms) + ($9,599.24 x bathrooms) + ($-17,421.84 x miles_to_school)

    pred_string = f"predicted price = ${B[0]:.2f}"
    for i in range(1,len(B)):
        pred_string += f" + (${B[i]:.2f} x {labels[i]})"
    return pred_string


# Return the R2 score for coefficients B
# Given inputs X and outputs Y
def score(B, X, Y):
    ## Your code here
    y_predict = np.dot(X,B)
    R2 = r2_score(y_true = Y, y_pred = y_predict)
    return R2
