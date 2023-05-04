import pandas as pd

series = pd.Series(["Volvo", "BMW", "Volvo",])

colours = (["red", "blue", "green"])

car_data =pd.DataFrame({"Car make": series, "Colour": colours})



cibo = pd.read_csv("ciao.csv")

#cibo.to_csv("ciao3.csv", index=False)

car_sales = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales.csv")

#car_sales.to_csv("car_sales.csv", index=False) 

"""
print(car_sales.dtypes)

print(car_sales.columns)

print(car_sales.index)

print(car_sales.describe())

print(car_sales.info())

print(car_sales["Doors"].sum())

print(car_sales["Doors"].mean())
 
# print(car_sales["Doors"]) -> print(car_sales.Doors)
"""


#SELECTING AND VIEWING DATA WITH PANDAS
"""

print(car_sales.head(3))

print(car_sales.tail(3))

animals = pd.Series(["dog", "cat", "bird", "rabbit", "mouse", "elephant"], index=[8,3,4,100,3,5])

print(animals.loc[3]) #cat e mouse

print(animals.iloc[3]) #rabbit, si riferisce alla posizione

print(animals.iloc[3:]) #"mouse", "elephant"

print(car_sales[car_sales.Make == "Honda"])

print(car_sales[car_sales["Odometer (KM)"] > 100000])
"""

#SELECTING AND VIEWING DATA WITH PANDAS PART2
"""
print(pd.crosstab(car_sales["Make"], car_sales["Doors"]))

print(car_sales.groupby(["Make"]).mean())

import matplotlib.pyplot as plt
#car_sales["Odometer (KM)"].plot()
#car_sales["Odometer (KM)"].hist()
#plt.show()

car_sales["Price"]= car_sales["Price"].str.replace('[\$\,\.]', '').astype(int) /100

car_sales["Price"].plot()

plt.show()
"""

#MANIPULATING DATA
"""
car_sales["Make"] =car_sales["Make"].str.lower()
# print(car_sales)

car_missing_data = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/car-sales-missing-data.csv")

# car_missing_data.to_csv("car_missing_data.csv", index=False)  

# car_missing_data.Odometer = car_missing_data.Odometer.fillna(car_missing_data.Odometer.mean())
# al posto di riassegnarla posso fare così ->

car_missing_data.Odometer.fillna(car_missing_data.Odometer.mean(), inplace=True)

# print(car_missing_data)

car_missing_data.dropna(inplace=True)

car_missing_data.to_csv("car_missing_data.csv", index=False) 

print(car_missing_data)
"""

#MANIPULATING DATA2
"""
seats_column = pd.Series([5,5,5,5,5])

car_sales["Seats"] = seats_column

car_sales.Seats.fillna(5, inplace=True)

#print(car_sales)

fuel_economy = [7.5,9.2,5.0,9.6,8.7,7.5,9.2,5.0,9.6,8.7]
car_sales["Fuel per 100km"] = fuel_economy

car_sales["Total fuel used"] = car_sales["Odometer (KM)"] / car_sales["Fuel per 100km"]

car_sales["Passed road safety"] = True

car_sales.drop("Colour", axis=1, inplace=True)
print(car_sales)

car_sales.to_csv("car_new_data.csv", index=False) 
"""
#MANIPULATING DATA2

"""
car_new_data = pd.read_csv("car_new_data.csv")

car_new_data_shuffled =car_new_data.sample(frac=1)

car_new_data_shuffled = car_new_data_shuffled.reset_index()

car_new_data["Odometer (KM)"] = car_new_data["Odometer (KM)"].apply(lambda x: x /1.6)
print(car_new_data)
"""