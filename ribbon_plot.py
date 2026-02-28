import numpy as np
import matplotlib.pyplot as plt
from distribution_fitting import DistributionQuantiles

def ribbon_plot(
    data,
    x_axis,
    predictions,
    actuals,
    reversion,
    dist_params,
    quantile_levels=[0.025, 0.25, 0.75, 0.975],
    monthly_median_func=None,
    show_connectors=True,
    title=None,
    ylim=None
):
    """
    Custom ribbon plot:
    1. Show ribbons for IQR and 95% CI
    2. Actual and predicted values as lines with markers
    3. Join prediction and observation with faded dotted lines (optional)
    4. Include monthly median price trend as a line
    """
    x = data[x_axis].values
    preds = data[predictions].values
    acts = data[actuals].values
    rev = data[reversion].values

    
    q_offsets = DistributionQuantiles.get_quantiles(dist_params, quantile_levels)
    q_lower_outer = preds + rev * q_offsets[0]
    q_lower_inner = preds + rev * q_offsets[1]
    q_upper_inner = preds + rev * q_offsets[2]
    q_upper_outer = preds + rev * q_offsets[3]

    plt.figure(figsize=(12, 6))
    # Ribbons
    plt.fill_between(x, q_lower_outer, q_upper_outer, color='royalblue', alpha=0.2, label='95% CI')
    plt.fill_between(x, q_lower_inner, q_upper_inner, color='royalblue', alpha=0.4, label='IQR')

    # Actual and predicted as lines with markers
    plt.plot(x, preds, marker='o', markersize=4, linestyle='-', color='blue', label='Prediction', alpha=0.5)
    plt.plot(x, acts, marker='o', markersize=4, linestyle='-', color='red', label='Actual', alpha=0.5)

    # Connectors
    if show_connectors:
        for xi, pi, ai in zip(x, preds, acts):
            plt.plot([xi, xi], [pi, ai], linestyle=':', color='gray', alpha=0.3)

    # Monthly median price trend
    if monthly_median_func is not None:
        monthly_median = monthly_median_func(data[actuals], lag_hours=30*24)
        plt.plot(x, monthly_median, color='green', linestyle='--', label='Monthly Median')

    plt.title(title or f"Ribbon Plot ({dist_params.name} Errors)", fontsize=12, fontweight='bold')
    plt.xlabel("Time / Index")
    plt.ylabel("Price")
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.show()
