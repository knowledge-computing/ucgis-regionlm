import logging
import joblib

class ModelTrainerBase:
    def __init__(self, data=None, target_column=None, verbose=False):
        self.data = data
        self.target_column = target_column
        self.model = None
        self.verbose = verbose

    def train_model(self):
        raise NotImplementedError

    def save_model(self, model_path=None, save=False):
        try:
            if self.model is not None:
                if save and model_path:
                    joblib.dump(self.model, model_path)
                    if self.verbose:
                        logging.info(f"Model saved to {model_path} successfully.")
        except Exception as e:
            logging.error(f"Error saving model to {model_path}: {e}")

    def load_model(self, model_path=None):
        try:
            if model_path:
                self.model = joblib.load(model_path)
                if self.verbose:
                    logging.info(f"Model loaded from {model_path} successfully.")
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")

    def predict(self, data):
        try:
            if self.model is not None and data is not None:
                return self.model.predict(data)
        except Exception as e:
            logging.error(f"Error predicting data: {e}")
        return None

    def get_model(self):
        return self.model
