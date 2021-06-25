# Importing the libraries
import matplotlib
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


# Reading the cleaned data
data = pd.read_csv("austin_final.csv")

X = data.drop(["PrecipitationSumInches"], axis=1)

Y = data["PrecipitationSumInches"]
Y = Y.values.reshape(-1, 1)

day_index = 798
days = [i for i in range(Y.size)]

#Training the model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=101
)

#Testing the data
clf = LinearRegression()
clf.fit(X_train, y_train)

# Printing out the coefficients
print(clf.coef_)

# Predicting the test data
predictions = clf.predict(X_test)

# Scatter plot
plt.scatter( predictions,y_test, color="blue")
plt.ylabel("PrecipitationSumInches")
plt.xlabel("Predicted")
plt.show()


# Evaluating the errors
print("MAE: {}".format(metrics.mean_absolute_error(y_test, predictions)))
print("MSE: {}".format(metrics.mean_squared_error(y_test, predictions)))
print("RMSE: {}".format(np.sqrt(metrics.mean_squared_error(y_test, predictions))))

inp = np.array(
    [
        [74],
        [60],
        [45],
        [67],
        [49],
        [43],
        [33],
        [45],
        [57],
        [29.68],
        [10],
        [7],
        [2],
        [0],
        [20],
        [4],
        [31],
    ]
)

inp = inp.reshape(1, -1)

# Printing the output
print("The precipitation in inches for the input is:", clf.predict(inp))

print("The precipitation trend graph: ")
plt.scatter(days, Y, color="orange")
plt.scatter(days[day_index], Y[day_index], color="red")
plt.title("Precipitation level")
plt.xlabel("Days")
plt.ylabel("Precipitation in inches")

# Ploting a graph of precipitation levels vs n# of days
plt.show()

# import seaborn as sns

# sns.set(color_codes=True)
# sns.lmplot(x=days, y="TempAvgF", data=data)

x_f = X.filter(
    [
        "TempAvgF",
        "DewPointAvgF",
        "HumidityAvgPercent",
        "SeaLevelPressureAvgInches",
        "VisibilityAvgMiles",
        "WindAvgMPH",
    ],
    axis=1,
)


print("Preciptiation Vs Selected Attributes Graph: ")
for i in range(x_f.columns.size):
    plt.subplot(3, 2, i + 1)
    plt.scatter(days, x_f[x_f.columns.values[i][:100]], color="g")
    plt.scatter(days[day_index], x_f[x_f.columns.values[i]][day_index], color="r")
    plt.xlabel("Days")
    plt.ylabel(x_f.columns[i])


# plot a graph with a few features vs precipitation to observe the trends
plt.show()


#Recreating the data frame with their resepctive coeffients
pd.DataFrame(clf.coef_ , X.columns, columns=['Coeffecient'])
