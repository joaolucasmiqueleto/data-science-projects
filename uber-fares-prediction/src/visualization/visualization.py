import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def scattering_true_predicted_values(
    y_test: np.array,
    model_predictions: np.array,
    rmse_metric: float,
    mae_metric: float,
    r2_metric: float,
):
    """This function gives a scatter plot relating the true values
    of target data and the predicted ones by the model.

    Args:
        y_test (np.array): true values of target data
        model_predictions (np.array): predictions of the model
        rmse_metric (float): root mean squared error metric
        mae_metric (float): mean absolute error metric
        r2_metric (float): r2 metric
    """

    mae = format(mae_metric, ".2f")
    rmse = format(rmse_metric, ".2f")
    r2 = format(r2_metric, ".2f")

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.set_style("darkgrid")
    plt.scatter(y_test, model_predictions, edgecolors="black", facecolors="none")
    plt.plot(y_test, y_test, "r")
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    text = "RMSE: " + str(rmse) + "  MAE: " + str(mae) + "   R2: " + str(r2)
    ax.set(xlabel="True values", ylabel="Predicted values")
    plt.title("Actual vs Predicted")
    plt.text(
        xmax - 0.01 * xmax,
        ymax - 0.01 * ymax,
        text,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=12,
    )
    plt.axis("scaled")
    plt.show()
