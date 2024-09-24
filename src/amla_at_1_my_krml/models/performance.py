def print_classification_scores(y_true, y_preds, y_proba, set_name=None):
    """Print the classification metrics for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score, classification_report
    from sklearn.metrics import roc_auc_score

    # Calculating the classification metrics
    accuracy_val = accuracy_score(y_true, y_preds)
    f1_val = f1_score(y_true, y_preds, average='weighted')
    recall_val = recall_score(y_true, y_preds, average='weighted')
    precision_val = precision_score(y_true, y_preds, average='weighted')

    # Calculating the AUROC score for the validation set
    roc_auc_val = roc_auc_score(y_true, y_proba)

    # Classification report
    report = classification_report(y_true, y_preds)

    # Print the results
    print("\n Classification metrics on the validation set:")
    print(f"accuracy score: {accuracy_val}")
    print(f"f1 score: {f1_val}" )
    print(f"recall score: {recall_val}")
    print(f"precision score: {precision_val}")
    print("\n AUROC Score of the validation set:")
    print(roc_auc_val)
    print('\n Classification Report')
    print('------------------')
    print(report)

def plot_auroc(y_true, y_proba):

    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    auroc_score = roc_auc_score(y_true, y_proba)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUROC = {auroc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--') 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def print_regression_scores(model_name, y_true, y_preds):
    """
    Print the regression metrics ( RMSE and MAE )for the provided data

    Parameters
    ----------
    model_name : str
        Name of the model to be evaluated
    y_true : Numpy Array
        Actual target
    y_preds : Numpy Array
        Predicted target

    Returns
    -------
    Nothing
    """
    from sklearn.metrics import mean_absolute_error as mae
    from sklearn.metrics import root_mean_squared_error as rmse

    print(f"{model_name} RMSE:")
    print(rmse(y_true, y_preds))
    print(f"{model_name} MAE:")
    print(mae(y_true, y_preds))

