import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

#load data
data_frame = pandas.read_csv("car details v4.csv")

#convert indian rupees to cad
rupee_to_cad = 0.01525
data_frame["Price"] = data_frame["Price"] * rupee_to_cad

#remove columns that aren't needed
data_frame = data_frame.drop(columns = ["Color", "Length", "Width", "Height", "Max Power", "Max Torque"])

#remove "cc" from engine data
cleaned_engine = []
for cc in data_frame["Engine"]:
    if pandas.notna(cc):
        cc = str(cc)
        cc = cc.replace("cc", "")
        cc = cc.strip()
        cleaned_engine.append(cc)
    else:
        cleaned_engine.append(cc)

data_frame["Engine"] = cleaned_engine

