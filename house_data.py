import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset.csv")

X = data[['Size','Bedrooms','Age']]
y = data['Price']

model = LinearRegression()

model.fit(X,y)

size = float(input("Enter house size: "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter house age: "))

prediction = model.predict([[size,bedrooms,age]])

print("Predicted House Price:", round(prediction[0]))

plt.scatter(data['Size'],data['Price'])
plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("House Price vs Size")
plt.show()