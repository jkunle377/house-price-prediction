from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df["PRICE"] = california.target
#print(df.head())
#print(df.describe())
#print(df.columns)
#df.info()

#plt.scatter(df["MedInc"], df["PRICE"], alpha=0.3)
#plt.xlabel("Median Income(100k $)")
#plt.ylabel("House Price (100k $)")
#plt.title("Income vs House Price")
#plt.savefig("california_price_plot.png")
#print("Plot saved as 'california_price_plot.png'")

X = df[["MedInc"]]
y = df["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, alpha=0.3, label="Actual")
plt.plot(X_test, y_pred, color="red", label="Predicted Line")
plt.xlabel("Median Income (100k $)")
plt.ylabel("House Price (100k$)")
plt.title("Linear Regression: Income vs Hosue Price")
plt.legend()
plt.tight_layout()
plt.savefig("regression_line.png")

