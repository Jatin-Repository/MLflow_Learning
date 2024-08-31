import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging
from ucimlrepo import fetch_ucirepo 


# logging.basicConfig(Level=logging.WARN)
# logger = logging.getLogger(__name__)

# logger = logging.FileHandler(filename='example.log', encoding='utf-8')
# logging.basicConfig(handlers=[logger], level=logging.DEBUG)\


logging.basicConfig(filename="example_file.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def eval_metrics(actual,pred):
    rmse = np.sqrt(mean_absolute_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # csv_url = (
    #     "https://github.com/aniruddhachoudhury/Red-Wine-Quality/blob/master/winequality-red.csv"
    # )
    
    try:
        # fetch dataset 
        data = fetch_ucirepo(id=186) 
        #print(data)
        # data = pd.read_csv(csv_url,sep=";")
    except Exception as e:
        logger.exception(
            "Unable tto download training and test CSV, check internet connection. Error: %s",e
        )
    

    train,test = data.data.features , data.data.targets 
    train_x,test_x,train_y,test_y = train_test_split(train,test,test_size=0.3,random_state=42)

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=42)
        lr.fit(train_x,train_y)

        predicted_qualities = lr.predict(test_x)
        #print(eval_metrics(test_y,predicted_qualities))
        (rmse,mae,r2) = eval_metrics(test_y,predicted_qualities)

        print("Elasticnet Model (alpha={:f}):".format(alpha,l1_ratio))
        print(" RMSE: %s" %rmse)
        print(" MAE: %s" %mae)
        print(" R2: %s"%r2)

        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric('mae',mae)

        # predictions = lr.predict(train_x)
        # signature = infer_signature(train_x,predictions)

        #For Remote Server only (Dagshub)

        remote_server_uri = "https://dagshub.com/Jatin-Repository/First_MLOps_Project.mlflow"
        mlflow.set_tracking_uri =(remote_server_uri)


        # remote sever url (AWS), default will save in local.
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme 

        if tracking_url_type_store != "file":
            # mlflow.sklearn.log_model(
            #     lr, "model", registered_model_name="ElasticnetWineModel", signature=signature
            # )
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel"
            )
        
        else:
            #mlflow.sklearn.log_model(lr,"model",signature=signature)
            mlflow.sklearn.log_model(lr,"model")
             

