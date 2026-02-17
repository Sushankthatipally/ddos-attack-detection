import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_alpha_vs_f1(tuning_results, metric_name):
    alphas = [r["alpha"] for r in tuning_results]
    f1_scores = [r["f1"] for r in tuning_results]

    plt.figure()
    plt.plot(alphas, f1_scores, marker='o')
    plt.xlabel("Alpha (α)")
    plt.ylabel("F1-score")
    plt.title(f"α vs F1-score ({metric_name})")
    plt.grid(True)
    plt.show()


def plot_roc_curve(y_true, scores, metric_name):
    """
    scores:
    - distance values for distance-based metrics
    - similarity values for cosine
    """

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({metric_name})")
    plt.legend()
    plt.grid(True)
    plt.show()
