"""
Paper Figure Generator for SimAM-KD
Author: Raju Kumar Yadav (itsmeerajuyadav@gmail.com)

Generates publication-quality figures for the paper:
1. Attention comparison bar chart
2. Training curves
3. Temperature ablation
4. Alpha ablation
5. Pruning accuracy vs parameters
6. Model comparison radar chart
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Use a clean style for papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
})

# Color palette
COLORS = {
    'none': '#1f77b4',
    'simam': '#ff7f0e',
    'ca': '#2ca02c',
    'parallel': '#d62728',
    'baseline': '#7f7f7f',
    'kd': '#9467bd',
    'pruned': '#8c564b',
}


def load_results(results_dir):
    """Load results from JSON file"""
    results_file = os.path.join(results_dir, 'all_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def fig1_attention_comparison(results, save_dir):
    """Figure 1: Attention mechanism comparison"""
    print("Generating Figure 1: Attention Comparison...")

    if 'attention_comparison' not in results:
        print("  Skipped - no attention comparison data")
        return

    data = results['attention_comparison']
    methods = list(data.keys())
    accuracies = [data[m]['accuracy'] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(methods, accuracies, color=[COLORS.get(m, '#333') for m in methods],
                  edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Attention Mechanism')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Effect of Attention Mechanisms on CIFAR-10')
    ax.set_ylim(80, max(accuracies) + 3)

    # Add baseline reference line
    if 'none' in data:
        ax.axhline(y=data['none']['accuracy'], color='gray', linestyle='--',
                   label=f"Baseline (no attention): {data['none']['accuracy']:.2f}%")
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig1_attention_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig1_attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig1_attention_comparison.pdf/png")


def fig2_temperature_ablation(results, save_dir):
    """Figure 2: Temperature ablation study"""
    print("Generating Figure 2: Temperature Ablation...")

    if 'temperature_ablation' not in results:
        print("  Skipped - no temperature ablation data")
        return

    data = results['temperature_ablation']
    temps = []
    accs = []
    for k, v in data.items():
        temps.append(v['temperature'])
        accs.append(v['accuracy'])

    # Sort by temperature
    sorted_idx = np.argsort(temps)
    temps = np.array(temps)[sorted_idx]
    accs = np.array(accs)[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(temps, accs, 'o-', color=COLORS['kd'], linewidth=2, markersize=10)

    # Highlight best
    best_idx = np.argmax(accs)
    ax.scatter([temps[best_idx]], [accs[best_idx]], color='red', s=200, zorder=5,
               marker='*', label=f'Best: T={temps[best_idx]:.0f} ({accs[best_idx]:.2f}%)')

    for t, a in zip(temps, accs):
        ax.annotate(f'{a:.2f}%', (t, a), textcoords="offset points",
                    xytext=(0, 10), ha='center')

    ax.set_xlabel('Temperature (T)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Effect of Distillation Temperature')
    ax.set_xticks(temps)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig2_temperature_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig2_temperature_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig2_temperature_ablation.pdf/png")


def fig3_alpha_ablation(results, save_dir):
    """Figure 3: Alpha ablation study"""
    print("Generating Figure 3: Alpha Ablation...")

    if 'alpha_ablation' not in results:
        print("  Skipped - no alpha ablation data")
        return

    data = results['alpha_ablation']
    alphas = []
    accs = []
    for k, v in data.items():
        alphas.append(v['alpha'])
        accs.append(v['accuracy'])

    sorted_idx = np.argsort(alphas)
    alphas = np.array(alphas)[sorted_idx]
    accs = np.array(accs)[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(range(len(alphas)), accs, color=COLORS['simam'], edgecolor='black', linewidth=1.2)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'α={a}' for a in alphas])

    for i, (a, acc) in enumerate(zip(alphas, accs)):
        ax.text(i, acc + 0.2, f'{acc:.2f}%', ha='center', fontweight='bold')

    ax.set_xlabel('Alpha (α) - Weight for Soft Labels')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Effect of Knowledge Distillation Alpha')
    ax.set_ylim(min(accs) - 2, max(accs) + 2)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig3_alpha_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig3_alpha_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_alpha_ablation.pdf/png")


def fig4_pruning_tradeoff(results, save_dir):
    """Figure 4: Pruning accuracy vs parameter reduction"""
    print("Generating Figure 4: Pruning Trade-off...")

    if 'pruning_ablation' not in results:
        print("  Skipped - no pruning ablation data")
        return

    data = results['pruning_ablation']
    param_reductions = []
    accuracies = []
    labels = []

    for k, v in data.items():
        if 'error' not in v:
            param_reductions.append(v['param_reduction'])
            accuracies.append(v['accuracy'])
            labels.append(k)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot points
    scatter = ax.scatter(param_reductions, accuracies, c=COLORS['pruned'],
                         s=150, edgecolors='black', linewidth=1.5, zorder=5)

    # Connect with line
    sorted_idx = np.argsort(param_reductions)
    pr_sorted = np.array(param_reductions)[sorted_idx]
    acc_sorted = np.array(accuracies)[sorted_idx]
    ax.plot(pr_sorted, acc_sorted, '--', color=COLORS['pruned'], alpha=0.5)

    # Add original (0% pruning) point
    if 'attention_comparison' in results and 'parallel' in results['attention_comparison']:
        orig_acc = results['attention_comparison']['parallel']['accuracy']
        ax.scatter([0], [orig_acc], c=COLORS['kd'], s=200, marker='s',
                   edgecolors='black', linewidth=1.5, zorder=6, label='Original (KD)')
        ax.plot([0] + list(pr_sorted), [orig_acc] + list(acc_sorted), '--',
                color='gray', alpha=0.3)

    # Labels
    for pr, acc, label in zip(param_reductions, accuracies, labels):
        ax.annotate(f'{label}\n{acc:.1f}%', (pr, acc), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=10)

    ax.set_xlabel('Parameter Reduction (%)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Accuracy vs. Model Compression')
    ax.legend(loc='lower left')
    ax.set_xlim(-5, max(param_reductions) + 10)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig4_pruning_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig4_pruning_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_pruning_tradeoff.pdf/png")


def fig5_main_results(results, save_dir):
    """Figure 5: Main results comparison"""
    print("Generating Figure 5: Main Results...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline results (hardcoded from earlier experiments)
    baseline_results = {
        'MobileNetV3\n(baseline)': 83.53,
        '+ Parallel\nAttention': 83.97,
        '+ Knowledge\nDistillation': 88.43,
        '+ Pruning\n(22% reduction)': 88.34,
        '+ Pruning\n(29% reduction)': 87.92,
    }

    # Override with actual results if available
    if 'attention_comparison' in results:
        if 'none' in results['attention_comparison']:
            baseline_results['MobileNetV3\n(baseline)'] = results['attention_comparison']['none']['accuracy']
        if 'parallel' in results['attention_comparison']:
            baseline_results['+ Parallel\nAttention'] = results['attention_comparison']['parallel']['accuracy']

    methods = list(baseline_results.keys())
    accs = list(baseline_results.values())

    colors = ['#7f7f7f', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    bars = ax.barh(methods, accs, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontweight='bold')

    ax.set_xlabel('Top-1 Accuracy (%)')
    ax.set_title('SimAM-KD: Progressive Improvement on CIFAR-10')
    ax.set_xlim(80, 92)

    # Add improvement annotations
    ax.axvline(x=83.53, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig5_main_results.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig5_main_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_main_results.pdf/png")


def fig6_cifar100(results, save_dir):
    """Figure 6: CIFAR-100 results"""
    print("Generating Figure 6: CIFAR-100 Results...")

    if 'cifar100' not in results:
        print("  Skipped - no CIFAR-100 data")
        return

    c100 = results['cifar100']

    fig, ax = plt.subplots(figsize=(6, 5))

    methods = ['Baseline', 'With KD']
    accs = [c100['baseline_accuracy'], c100['kd_accuracy']]

    bars = ax.bar(methods, accs, color=[COLORS['baseline'], COLORS['kd']],
                  edgecolor='black', linewidth=1.2, width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', fontweight='bold')

    # Show improvement
    improvement = c100['improvement']
    ax.annotate(f'+{improvement:.2f}%', xy=(1, accs[1]), xytext=(1.3, accs[1] - 3),
                fontsize=14, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))

    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('CIFAR-100 Results')
    ax.set_ylim(0, max(accs) + 10)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fig6_cifar100.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/fig6_cifar100.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: fig6_cifar100.pdf/png")


def generate_table_latex(results, save_dir):
    """Generate LaTeX tables for the paper"""
    print("Generating LaTeX tables...")

    tables = []

    # Table 1: Main Results
    table1 = r"""
\begin{table}[h]
\centering
\caption{Main Results on CIFAR-10}
\label{tab:main_results}
\begin{tabular}{lccc}
\toprule
Method & Accuracy (\%) & Params (M) & Improvement \\
\midrule
MobileNetV3-S (Baseline) & 83.53 & 1.68 & - \\
+ Parallel Attention & 83.97 & 1.78 & +0.44 \\
+ Knowledge Distillation & 88.43 & 1.78 & +4.46 \\
+ Pruning (22\%) & 88.34 & 1.38 & +4.37 \\
+ Pruning (29\%) & 87.92 & 1.27 & +3.95 \\
\bottomrule
\end{tabular}
\end{table}
"""
    tables.append(('table1_main_results.tex', table1))

    # Table 2: Attention Comparison
    if 'attention_comparison' in results:
        rows = []
        for att, data in results['attention_comparison'].items():
            rows.append(f"{att.capitalize()} & {data['accuracy']:.2f} & {data['params']/1e6:.2f} \\\\")

        table2 = r"""
\begin{table}[h]
\centering
\caption{Attention Mechanism Comparison}
\label{tab:attention}
\begin{tabular}{lcc}
\toprule
Attention Type & Accuracy (\%) & Params (M) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
        tables.append(('table2_attention.tex', table2))

    # Save tables
    for filename, content in tables:
        with open(f'{save_dir}/{filename}', 'w') as f:
            f.write(content)
        print(f"  Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Directory with experiment results')
    parser.add_argument('--output-dir', type=str, default='./figures',
                       help='Directory to save figures')
    args = parser.parse_args()

    print("=" * 60)
    print("PAPER FIGURE GENERATOR")
    print("Author: Raju Kumar Yadav")
    print("=" * 60)

    # Find latest results directory if not specified
    if args.results_dir is None:
        results_dirs = sorted(Path('./results').glob('complete_*'))
        if results_dirs:
            args.results_dir = str(results_dirs[-1])
            print(f"Using latest results: {args.results_dir}")
        else:
            # Try proper results
            results_dirs = sorted(Path('./results').glob('proper_*'))
            if results_dirs:
                args.results_dir = str(results_dirs[-1])
                print(f"Using proper results: {args.results_dir}")
            else:
                print("No results found! Run experiments first.")
                return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving figures to: {args.output_dir}")

    # Load results
    results = load_results(args.results_dir)
    if results is None:
        print(f"Could not load results from {args.results_dir}")
        # Use hardcoded results from the proper run
        results = {
            'attention_comparison': {
                'parallel': {'accuracy': 88.43, 'params': 1780000}
            },
            'pruning_ablation': {
                '15%': {'accuracy': 88.34, 'param_reduction': 22.2},
                '20%': {'accuracy': 87.92, 'param_reduction': 28.8},
            }
        }
        print("Using hardcoded results from previous run")

    # Generate figures
    fig1_attention_comparison(results, args.output_dir)
    fig2_temperature_ablation(results, args.output_dir)
    fig3_alpha_ablation(results, args.output_dir)
    fig4_pruning_tradeoff(results, args.output_dir)
    fig5_main_results(results, args.output_dir)
    fig6_cifar100(results, args.output_dir)

    # Generate LaTeX tables
    generate_table_latex(results, args.output_dir)

    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED!")
    print(f"Location: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
