from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn. metrics import mean_squared_error
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso
import seaborn as sns
import numpy as np
import joblib

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df["PRICE"] = california.target

corr_matrix = df.drop("PRICE", axis=1).corr().abs() #Get correlation matrix of just input features(not price)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) #mask the upper triangle to avoid duplicate pairs
corr_matrix_lower = corr_matrix.mask(mask)
high_corr_pairs = corr_matrix_lower.stack().loc[lambda x: x > 0.8]
#print("Highly correlated feature pairs (correlation > 0.80:")
#print(high_corr_pairs)

ax = sns.heatmap(df.corr(), annot=True) #plotting the heatmap for correlation
#plt.savefig("correlation.png")
#print(df.head())
#print(df.describe())
#print(df.columns)
#df.info()

X = df.drop(columns=["AveBedrms", "Latitude", "PRICE"])
y = df["PRICE"]
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

models = {
 "Linear Regression": LinearRegression(),
 "Ridge Regression": Ridge(alpha=10),
 "Lasso Re gressioon": Lasso(alpha=0.001)
}

for name, model in models.items():
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
#model = LinearRegression()
#model.fit(X_train, y_train)

#y_pred = model.predict(X_test)

#ridgeReg = Ridge(alpha=100)
#ridgeReg.fit(X_train, y_train)

#y_pred = ridgeReg.predict(X_test)

#lasso = Lasso(alpha=0.001)
#lasso.fit(X_train, y_train)

#y_pred = lasso.predict(X_test)

 r2 = r2_score(y_test, y_pred)
 mae = mean_absolute_error(y_test, y_pred)
 mse = mean_squared_error(y_test, y_pred)
 rmse = np.sqrt(mse)

 print(f"{name}:")
 print(f" R2 Score  :{r2:.4f}")
 print(f" MAE    :{mae:.4f}")
 print(f" MSE    :{mse:.4f}")
 print(f" RMSE    :{rmse:.4f}")
 print(" Cofficients :")
for feat, coef in zip(feature_names, model.coef_):
 print(f" {feat}: {coef:.4f}")

joblib.dump(model, "house_price_model.pkl") #save the model
loaded_model = joblib.load("house_price_model.pkl") #load the model
new_prediction = loaded_model.predict(X_test)
print("Prediction from loaded model:", new_prediction[:5])

#print('r2 score for this model is', r2)
#print('mean absolute error', mae)#
#print('mean squared error', mse)
#print('Model Coefficients', lasso.coef_)

#plt.figure(figsize=(8,6))
#plt.scatter(X_test["AveRooms"], y_test, alpha=0.3, label="Actual")
#plt.plot(X_test["AveRooms"], y_pred, color="red", label="Predicted Line")
#plt.xlabel("AHA (100k $)")
#plt.ylabel("House Price (100k$)")
#plt.title("Linear Regression: AHA vs House Price")
#plt.legend()
#plt.tight_layout()
#plt.savefig("regression_line_11.png")

