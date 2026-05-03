from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    data = load_iris()
    X, y = data.data, data.target
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    joblib.dump(model, "../model/iris_model.pkl")

if __name__ == "__main__":
    train_model()