import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor


# =========================
# Load dataset
# =========================
data = pd.read_csv("plant_stress_dataset_50000_rows.csv")

# Features
X = data[['Frequency','Phase','VPD','VWC','Weight']]

# Targets
y_imp = data['Impedance']
y_rh = data['RH']
y_temp = data['Temperature']


# =========================
# Normalization
# =========================
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# =========================
# Train Test Split (70/30)
# =========================
X_train, X_test, y_imp_train, y_imp_test = train_test_split(
    X, y_imp, test_size=0.3, random_state=42)

_, _, y_rh_train, y_rh_test = train_test_split(
    X, y_rh, test_size=0.3, random_state=42)

_, _, y_temp_train, y_temp_test = train_test_split(
    X, y_temp, test_size=0.3, random_state=42)


# =========================
# Models (Same as paper)
# =========================

MLP = MLPRegressor(
    hidden_layer_sizes=(20,30),
    max_iter=200,
    alpha=0.001,
    solver='adam',
    random_state=42
)

MLR = LinearRegression()

DT = DecisionTreeRegressor(random_state=42)

Ada = AdaBoostRegressor(
    n_estimators=100,
    random_state=42
)

KNN = KNeighborsRegressor(n_neighbors=2)

# Proposed AdapTree (DT + AdaBoost)
AdapTree = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=100,
    random_state=42
)

models = {
"MLP":MLP,
"MLR":MLR,
"DT":DT,
"AdaBoost":Ada,
"KNN":KNN,
"Proposed":AdapTree
}


# =========================
# Evaluation Function
# =========================

def evaluate(model,X_train,X_test,y_train,y_test):

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,pred)
    mse = mean_squared_error(y_test,pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,pred)

    pcc,_ = pearsonr(y_test,pred)

    return r2,mse,rmse,mae,pcc


# =========================
# Run models
# =========================

targets = {
"Impedance":(y_imp_train,y_imp_test),
"RH":(y_rh_train,y_rh_test),
"Temperature":(y_temp_train,y_temp_test)
}

for target,(y_train,y_test) in targets.items():

    print("\n===============================")
    print("Results for",target)
    print("===============================")

    for name,model in models.items():

        r2,mse,rmse,mae,pcc = evaluate(model,X_train,X_test,y_train,y_test)

        print(name)
        print("R2 =",r2)
        print("MSE =",mse)
        print("RMSE =",rmse)
        print("MAE =",mae)
        print("Pearson =",pcc)
        print("--------------------")