import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BootstrapMetricsCalculator:
    """Class to calculate some important regression metrics 
    from bootstrap methods applied to the test data
    """
    def __init__(self):
        self.bootstrapped_rmse_values = []
        self.bootstrapped_mae_values = []
        self.bootstrapped_r2_values = []

    def calculating_bootstrapped_metrics(
        self, y_test: np.array, model_predictions: np.array, bootstrap_iterations: int
    )->list:
        """This function implements a bootstrap method in the test data 
        calculating a list of values for each regression metric
        """
        random_number_generator = np.random.RandomState(seed=1234)
        index = np.arange(y_test.shape[0])

        for _ in range(bootstrap_iterations):
            pred_index = random_number_generator.choice(
                index, size=index.shape[0], replace=True
            )

            rmse_bootstrap = np.sqrt(
                mean_squared_error(y_test[pred_index], model_predictions[pred_index])
            )

            mae_bootstrap = mean_absolute_error(
                y_test[pred_index], model_predictions[pred_index]
            )

            r2_bootstrap = r2_score(y_test[pred_index], model_predictions[pred_index])

            self.bootstrapped_rmse_values.append(rmse_bootstrap)
            self.bootstrapped_mae_values.append(mae_bootstrap)
            self.bootstrapped_r2_values.append(r2_bootstrap)

    def get_mean_rmse_metric(self)->float:
        """This function receives a list of rmse values calculated
        from bootstrap and calculates the mean value of the list

        Returns:
            float: mean value of rmse list of values
        """
        return np.mean(self.bootstrapped_rmse_values)

    def get_mean_mae_metric(self)->float:
        """This function receives a list of mae values calculated
        from bootstrap and calculates the mean value of the list

        Returns:
            float: mean value of mae list of values
        """
        return np.mean(self.bootstrapped_mae_values)

    def get_mean_r2_metric(self)->float:
        """This function receives a list of r2 values calculated
        from bootstrap and calculates the mean value of the list

        Returns:
            float: mean value of r2 list of values
        """
        return np.mean(self.bootstrapped_r2_values)
