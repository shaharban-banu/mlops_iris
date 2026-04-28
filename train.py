import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("Starting training with MLflow...")

mlflow.set_tracking_uri("file:./mlruns")
# 🔥 create experiment
mlflow.set_experiment("iris_project")

# load data
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)

    # ✅ MLflow logging
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model", "RandomForest")

    # # 🔥 LOG MODEL (VERY IMPORTANT)
    # mlflow.sklearn.log_model(name= "model")

    # save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

print("MLflow logging done ✅")

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# import pandas as pd
# from sklearn.metrics import accuracy_score
# import mlflow
# import pickle

# mlflow.set_experiment("iris_project")

# data=pd.read_csv('data/Iris.csv')
# x=data.drop(columns=['Species'])
# y=data['Species']

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# with mlflow.start_run():
#     model=RandomForestClassifier()
#     model.fit(x_train,y_train)

#     pred=model.predict(x_test)
#     acc=accuracy_score(y_test,pred)

#     mlflow.log_metric("accuracy",acc)
#     mlflow.log_param("model","randomforest")

#     with open("model.pkl","wb") as f:
#         pickle.dump(model,f)

# print("model trained... accuracy ",acc)