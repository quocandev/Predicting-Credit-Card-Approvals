# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the dataset
cc_apps = pd.read_csv("cc_approvals.data", header=None) 
cc_apps.head()

# Giá trị missing
missing_value = ['0']

# Các giá trị bạn muốn thử cho từng tham số
tol_values = [1e-4, 1e-3, 1e-2]           # Độ dung sai (tolerance)
max_iter_values = [100, 200, 300]         # Số vòng lặp tối đa
C_values = [0.01, 0.1, 1, 10]             # Hệ số regularization
penalty_values = ['l1', 'l2']             # Loại regularization

# Chuyển missing value thành nan
cc_apps_replace = cc_apps.replace(missing_value, np.nan)

# copy
apps_input = cc_apps_replace.copy()

#xử lý missing value
for col in apps_input.columns:
    if apps_input[col].dtype == 'object':
        most_freq = apps_input[col].value_counts().index[0] #Dùng gtri xuất hiện nhiều nhất
        apps_input[col].fillna(most_freq, inplace=True) # Hàm fillna là hàm thay thế giá trị nan thành giá trị most_freq
    else:
        mean_value = apps_input[col].mean() # Dùng giá trị trung bình mean(), trung vị median() cho dữ liệu numeric
        apps_input[col].fillna(mean_value, inplace=True)

# Biến đổi dạng chuỗi phân loại thành nhị phân (0, 1)
apps_input = pd.get_dummies(apps_input, drop_first=True)

#Hàm iloc dùng để trích xuất dữ liệu
X = apps_input.iloc[:, :-1].values
y = apps_input.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

scaler = StandardScaler()

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train_scaler, y_train)

y_pred = model.predict(X_test_scaler)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)

acc = (cm[1][1] + cm[0][0]) / (cm[1][1] + cm[0][0] + cm[1][0] + cm[0][1])
precision = cm[1][1] / (cm[1][1] + cm[0][1])
recall = cm[1][1] / (cm[1][1] + cm[1][0])
f1_score = 2 * ((precision*recall)/(precision+recall))

# Dsach tham so
param_grid = dict(
    tol=tol_values,
    max_iter=max_iter_values,
    C=C_values,
    penalty=penalty_values
)

# Lưới test
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_scaler, y_train)

best_model = grid_search.best_estimator_

best_score = best_model.score(X_test_scaler, y_test)

print(best_score)








