import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2021)

print("Training set has n_rows=%d n_cols=%d" %(x_train.shape))
print("Test set has n_rows=%d n_cols=%d" %(x_test.shape))

train_data = "train_data.txt"
test_data = "test_data.txt"
train_labels = "train_labels.txt"
test_labels = "test_labels.txt"

# Save all datasets in col-major format
np.savetxt(train_data, x_train.T,fmt='%.7f')
np.savetxt(test_data, x_test.T,fmt='%.7f')
np.savetxt(train_labels, y_train,fmt='%.7f')
np.savetxt(test_labels, y_test,fmt='%.7f')