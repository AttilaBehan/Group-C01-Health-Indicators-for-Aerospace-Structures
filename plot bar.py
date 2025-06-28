import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

fitness_scores=pd.read_csv(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\fitness_scores.csv")
feature_names=pd.read_csv(r"C:\Users\attil\OneDrive\TU_Delft\C01_main\Feature_Names.csv")
for cols in fitness_scores.columns:
    fitness_list= fitness_scores[cols].dropna().apply(ast.literal_eval).tolist()
    feature_list = np.arange(len(fitness_list))
    #feature_list = feature_names[cols].dropna().tolist()

    trendability = [feat[0] for feat in fitness_list]
    monotonicity = [feat[1] for feat in fitness_list]
    prognosability = [feat[2] for feat in fitness_list]

    total_heights = [t + m + p for t, m, p in zip(trendability, monotonicity, prognosability)]
    mu = np.mean(total_heights)


    # X locations
    x = np.arange(len(fitness_list))

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width=0.4 

    ax.bar(x, prognosability, width=bar_width, label='Prognosability', color='lightgreen')
    ax.bar(x, monotonicity, width=bar_width, bottom=prognosability, label='Monotonicity', color='salmon')
    bottom_stack = [p + m for p, m in zip(prognosability, monotonicity)]
    ax.bar(x, trendability, width=bar_width, bottom=bottom_stack, label='Trendability', color='skyblue')

    ax.axhline(mu, color='black', linestyle='--', linewidth=1.5, label=f'μ = {mu:.2f}')
    ax.text(len(x) - 0.2, mu -0.05, 'μ', color='red', fontsize=16, va='bottom')

    # Formatting
    ax.set_xlim(-0.5, len(x) - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_list, rotation=0, ha='center')
    ax.set_ylabel('Fitness Score')
    ax.set_xlabel('Feature Index')
    ax.set_title(f'{cols} Feature Scores')
    ax.set_ylim(0, 3)
    ax.legend()

    plt.tight_layout()
    plt.show()