# Dependencies - matlab python3 numpy
#
# plotter python script to look at 2 dimentional data from kalman filters
# and analyse the result visually.
#
# running instuctions - $ python plotter.py


import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

file_location = "C:\\Users\\apoorva\\Desktop\\gitlab\\KalmanFilter\\testing\\measure4.txt"
f = open(file_location,'r')
read_data = f.read()

print ("data read!")

arr_data = read_data.split()

num_Lines = len(arr_data)

linestx = []
linestv = []
enestx = []
enestv = []
z = []
linupx = []
linupv = []
enupx = []
enupv = []
x = []

print ("sumber of lines " + str(num_Lines))

for i in range(num_Lines):
    j = i % 9

    if (j == 0):
        linestx.append(float(arr_data[i]))
    if (j == 1):
        linestv.append(float(arr_data[i]))
    if (j == 2):
        enestx.append(float(arr_data[i]))
    if (j == 3):
        enestv.append(float(arr_data[i]))
    if (j == 4):
        z.append(float(arr_data[i]))
    if (j == 5):
        linupx.append(float(arr_data[i]))
    if (j == 6):
        linupv.append(float(arr_data[i]))
    if (j == 7):
        enupx.append(float(arr_data[i]))
    if (j == 8):
       enupv.append(float(arr_data[i]))
       x.append(i/8 - 1)

# LKF_est = np.array([linestx])
# LKF_up = np.array([linupx])
# EnKF_est = np.array([enestx])
# EnKF_up = np.array([enupx])
# Measurements = np.array([z])



plt.plot (x, linestx,label='LKF_est')
plt.plot (x, linupx, label='LKF_up')
plt.plot (x, enestx, label='EnKF_est')
plt.plot (x, enupx, label='EnKF_up')
plt.plot (x, z, 'g^', label='Measurements')
plt.legend(loc='best')
plt.show()
