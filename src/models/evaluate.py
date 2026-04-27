from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_test, y_pred):
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'classification_report': classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }


def compare_metrics(metrics_a, metrics_b, name_a='model_a', name_b='model_b'):
    return {
        name_a: {
            'accuracy': metrics_a['accuracy'],
            'precision': metrics_a['precision'],
            'recall': metrics_a['recall'],
            'f1': metrics_a['f1'],
        },
        name_b: {
            'accuracy': metrics_b['accuracy'],
            'precision': metrics_b['precision'],
            'recall': metrics_b['recall'],
            'f1': metrics_b['f1'],
        },
    }
