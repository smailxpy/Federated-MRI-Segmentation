#!/usr/bin/env python3
"""
Visualization Tools for Federated Continual Learning
Team: 314IV | Topic: #6 Federated Continual Learning for MRI Segmentation

This module provides comprehensive visualization tools for analyzing experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ExperimentVisualizer:
    """Visualization tools for federated continual learning experiments"""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        # Load data
        self.metrics_data = self._load_metrics_data()
        self.forgetting_data = self._load_forgetting_data()
        self.experiment_report = self._load_experiment_report()

    def _load_metrics_data(self) -> pd.DataFrame:
        """Load metrics log data"""
        metrics_path = self.results_dir / "logs" / "metrics_log.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                data = json.load(f)

            # Convert to DataFrame
            records = []
            for entry in data:
                record = {
                    'round': entry['round'],
                    'client_id': entry['client_id'],
                    'task_id': entry['task_id'],
                    'timestamp': entry['timestamp']
                }
                record.update(entry['metrics'])
                records.append(record)

            return pd.DataFrame(records)
        return pd.DataFrame()

    def _load_forgetting_data(self) -> pd.DataFrame:
        """Load forgetting metrics data"""
        forgetting_path = self.results_dir / "logs" / "forgetting_log.json"
        if forgetting_path.exists():
            with open(forgetting_path, 'r') as f:
                data = json.load(f)

            # Convert to DataFrame
            records = []
            for entry in data:
                record = {
                    'task_id': entry['task_id'],
                    'timestamp': entry['timestamp']
                }
                record.update(entry['forgetting_rates'])
                records.append(record)

            return pd.DataFrame(records)
        return pd.DataFrame()

    def _load_experiment_report(self) -> Dict:
        """Load experiment report"""
        report_path = self.results_dir / "final_report.json"
        if report_path.exists():
            with open(report_path, 'r') as f:
                return json.load(f)
        return {}

    def plot_metrics_over_rounds(self, metrics: List[str] = None):
        """Plot metrics evolution over federated rounds"""
        if self.metrics_data.empty:
            print("❌ No metrics data available")
            return

        if metrics is None:
            metrics = ['dice_mean', 'hd95_mean', 'precision', 'recall']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            if metric in self.metrics_data.columns:
                ax = axes[i]

                # Group by round and calculate mean
                round_metrics = self.metrics_data.groupby('round')[metric].agg(['mean', 'std']).reset_index()

                # Plot with error bars
                ax.errorbar(round_metrics['round'], round_metrics['mean'],
                           yerr=round_metrics['std'], capsize=3, marker='o')

                ax.set_title(f'{metric.replace("_", " ").title()} Over Rounds')
                ax.set_xlabel('Round')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "metrics_over_rounds.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Metrics over rounds plot saved")

    def plot_task_performance(self):
        """Plot performance across continual learning tasks"""
        if self.metrics_data.empty:
            print("❌ No metrics data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metrics = ['dice_mean', 'hd95_mean', 'precision', 'recall']

        for i, metric in enumerate(metrics):
            if metric in self.metrics_data.columns:
                ax = axes[i]

                # Group by task and calculate statistics
                task_metrics = self.metrics_data.groupby('task_id')[metric].agg(['mean', 'std', 'min', 'max']).reset_index()

                # Plot box plot for each task
                self.metrics_data.boxplot(column=metric, by='task_id', ax=ax, grid=False)

                ax.set_title(f'{metric.replace("_", " ").title()} by Task')
                ax.set_xlabel('Task ID')
                ax.set_ylabel(metric.replace("_", " ").title())

        plt.suptitle('Task-wise Performance Distribution')
        plt.tight_layout()
        plt.savefig(self.plots_dir / "task_performance.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Task performance plot saved")

    def plot_forgetting_analysis(self):
        """Plot catastrophic forgetting analysis"""
        if self.forgetting_data.empty:
            print("❌ No forgetting data available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot forgetting rates over tasks
        forgetting_cols = [col for col in self.forgetting_data.columns if 'forgetting' in col and col != 'avg_forgetting_rate']

        if forgetting_cols:
            # Melt data for plotting
            forgetting_melted = self.forgetting_data.melt(
                id_vars=['task_id'],
                value_vars=forgetting_cols,
                var_name='metric_type',
                value_name='forgetting_rate'
            )

            sns.lineplot(data=forgetting_melted, x='task_id', y='forgetting_rate',
                        hue='metric_type', ax=ax1, marker='o')

            ax1.set_title('Forgetting Rates Over Tasks')
            ax1.set_xlabel('Current Task')
            ax1.set_ylabel('Forgetting Rate')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

        # Plot average forgetting rate
        if 'avg_forgetting_rate' in self.forgetting_data.columns:
            ax2.plot(self.forgetting_data['task_id'], self.forgetting_data['avg_forgetting_rate'],
                    marker='s', linewidth=2, markersize=8)

            ax2.set_title('Average Forgetting Rate')
            ax2.set_xlabel('Task ID')
            ax2.set_ylabel('Average Forgetting Rate')
            ax2.grid(True, alpha=0.3)

            # Add threshold line
            ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (10%)')
            ax2.legend()

        plt.tight_layout()
        plt.savefig(self.plots_dir / "forgetting_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Forgetting analysis plot saved")

    def plot_client_performance_distribution(self):
        """Plot performance distribution across clients"""
        if self.metrics_data.empty:
            print("❌ No metrics data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metrics = ['dice_mean', 'hd95_mean', 'precision', 'recall']

        for i, metric in enumerate(metrics):
            if metric in self.metrics_data.columns:
                ax = axes[i]

                # Violin plot for each client
                sns.violinplot(data=self.metrics_data, x='client_id', y=metric, ax=ax)

                ax.set_title(f'{metric.replace("_", " ").title()} Distribution by Client')
                ax.set_xlabel('Client ID')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "client_performance_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Client performance distribution plot saved")

    def plot_convergence_analysis(self):
        """Plot training convergence analysis"""
        if self.metrics_data.empty:
            print("❌ No metrics data available")
            return

        # Calculate convergence metrics
        convergence_data = []

        for client_id in self.metrics_data['client_id'].unique():
            client_data = self.metrics_data[self.metrics_data['client_id'] == client_id]

            for task_id in client_data['task_id'].unique():
                task_data = client_data[client_data['task_id'] == task_id]

                if len(task_data) > 1:
                    # Calculate convergence rate (improvement per round)
                    initial_metric = task_data[task_data['round'] == task_data['round'].min()]['dice_mean'].iloc[0]
                    final_metric = task_data[task_data['round'] == task_data['round'].max()]['dice_mean'].iloc[0]

                    rounds_taken = task_data['round'].max() - task_data['round'].min()
                    convergence_rate = (final_metric - initial_metric) / max(rounds_taken, 1)

                    convergence_data.append({
                        'client_id': client_id,
                        'task_id': task_id,
                        'initial_dice': initial_metric,
                        'final_dice': final_metric,
                        'convergence_rate': convergence_rate,
                        'rounds_to_converge': rounds_taken
                    })

        if convergence_data:
            conv_df = pd.DataFrame(convergence_data)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Convergence rates
            sns.barplot(data=conv_df, x='client_id', y='convergence_rate', ax=ax1)
            ax1.set_title('Convergence Rates by Client')
            ax1.set_xlabel('Client ID')
            ax1.set_ylabel('Dice Improvement per Round')
            ax1.tick_params(axis='x', rotation=45)

            # Rounds to converge
            sns.boxplot(data=conv_df, x='task_id', y='rounds_to_converge', ax=ax2)
            ax2.set_title('Rounds to Converge by Task')
            ax2.set_xlabel('Task ID')
            ax2.set_ylabel('Rounds to Converge')

            plt.tight_layout()
            plt.savefig(self.plots_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
            print("✓ Convergence analysis plot saved")

    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        if not self.experiment_report:
            print("❌ No experiment report available")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Metrics summary
        metrics_summary = self.experiment_report.get('metrics_summary', {})

        # Extract key metrics
        dice_mean = metrics_summary.get('dice_mean_mean', 0)
        dice_std = metrics_summary.get('dice_std_mean', 0)
        hd95_mean = metrics_summary.get('hd95_mean_mean', 0)
        hd95_std = metrics_summary.get('hd95_std_mean', 0)

        # Plot 1: Dice Coefficient
        axes[0, 0].bar(['Dice Mean'], [dice_mean], yerr=[dice_std], capsize=5, color='skyblue')
        axes[0, 0].set_title('Dice Coefficient Performance')
        axes[0, 0].set_ylabel('Dice Score')
        axes[0, 0].set_ylim(0, 1)

        # Plot 2: HD95
        axes[0, 1].bar(['HD95 Mean'], [hd95_mean], yerr=[hd95_std], capsize=5, color='lightcoral')
        axes[0, 1].set_title('Hausdorff Distance 95')
        axes[0, 1].set_ylabel('HD95 (mm)')

        # Plot 3: Forgetting analysis
        forgetting_summary = self.experiment_report.get('forgetting_summary', {})
        avg_forgetting = forgetting_summary.get('avg_forgetting_rate', 0)

        axes[0, 2].bar(['Avg Forgetting'], [avg_forgetting], color='lightgreen')
        axes[0, 2].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Target (<10%)')
        axes[0, 2].set_title('Catastrophic Forgetting')
        axes[0, 2].set_ylabel('Forgetting Rate')
        axes[0, 2].legend()

        # Plot 4: Experiment overview
        overview_data = {
            'Total Rounds': self.experiment_report.get('total_rounds', 0),
            'Total Clients': self.experiment_report.get('total_clients', 0),
            'Total Tasks': self.experiment_report.get('total_tasks', 0)
        }

        axes[1, 0].bar(overview_data.keys(), overview_data.values(), color='orange')
        axes[1, 0].set_title('Experiment Overview')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 5: Task progression (if available)
        if not self.metrics_data.empty:
            task_progression = self.metrics_data.groupby('task_id')['dice_mean'].mean().reset_index()
            axes[1, 1].plot(task_progression['task_id'], task_progression['dice_mean'],
                           marker='o', linewidth=2, markersize=8)
            axes[1, 1].set_title('Task-wise Performance')
            axes[1, 1].set_xlabel('Task ID')
            axes[1, 1].set_ylabel('Average Dice')
            axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Client distribution
        if not self.metrics_data.empty:
            client_perf = self.metrics_data.groupby('client_id')['dice_mean'].mean().reset_index()
            axes[1, 2].bar(client_perf['client_id'], client_perf['dice_mean'], color='purple')
            axes[1, 2].set_title('Client Performance Distribution')
            axes[1, 2].set_xlabel('Client ID')
            axes[1, 2].set_ylabel('Average Dice')
            axes[1, 2].tick_params(axis='x', rotation=45)

        plt.suptitle('Federated Continual Learning - Experiment Summary Dashboard', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Summary dashboard saved")

    def generate_experiment_report(self) -> str:
        """Generate a comprehensive text report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FEDERATED CONTINUAL LEARNING EXPERIMENT REPORT")
        report_lines.append("=" * 80)

        report_lines.append(f"\nTeam: 314IV")
        report_lines.append(f"Topic: #6 Federated Continual Learning for MRI Segmentation")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if self.experiment_report:
            report_lines.append(f"\n📊 EXPERIMENT OVERVIEW:")
            report_lines.append(f"• Total Rounds: {self.experiment_report.get('total_rounds', 'N/A')}")
            report_lines.append(f"• Total Clients: {self.experiment_report.get('total_clients', 'N/A')}")
            report_lines.append(f"• Total Tasks: {self.experiment_report.get('total_tasks', 'N/A')}")

            # Metrics summary
            metrics_summary = self.experiment_report.get('metrics_summary', {})
            report_lines.append(f"\n🎯 PERFORMANCE METRICS:")
            dice = metrics_summary.get('dice_mean_mean')
            hd95 = metrics_summary.get('hd95_mean_mean')
            precision = metrics_summary.get('precision_mean')
            recall = metrics_summary.get('recall_mean')
            if dice is not None:
                report_lines.append(f"Dice (mean): {dice:.4f}")
            if hd95 is not None:
                report_lines.append(f"HD95 (mean): {hd95:.4f} mm")
            if precision is not None:
                report_lines.append(f"Precision (mean): {precision:.4f}")
            if recall is not None:
                report_lines.append(f"Recall (mean): {recall:.4f}")
            # Forgetting summary
            forgetting_summary = self.experiment_report.get('forgetting_summary', {})
            report_lines.append(f"\n🧠 FORGETTING ANALYSIS:")
            if 'avg_forgetting_rate' in forgetting_summary:
                forgetting_rate = forgetting_summary['avg_forgetting_rate']
                status = "✓ GOOD" if forgetting_rate < 0.1 else "⚠️  HIGH"
                report_lines.append(f"Forgetting Rate: {forgetting_rate:.4f} ({status})")
        report_lines.append(f"\n📁 Results saved to: {self.results_dir}")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Save report
        report_path = self.results_dir / "experiment_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)

        print("✓ Experiment report generated")
        return report_text


def compare_experiments(exp_dirs: List[Path], exp_names: List[str]):
    """Compare multiple experiments"""
    if len(exp_dirs) != len(exp_names):
        print("❌ Mismatch between experiment directories and names")
        return

    comparison_data = []

    for exp_dir, exp_name in zip(exp_dirs, exp_names):
        visualizer = ExperimentVisualizer(exp_dir)

        if visualizer.experiment_report:
            metrics = visualizer.experiment_report.get('metrics_summary', {})
            forgetting = visualizer.experiment_report.get('forgetting_summary', {})

            comparison_data.append({
                'experiment': exp_name,
                'dice_mean': metrics.get('dice_mean_mean', 0),
                'hd95_mean': metrics.get('hd95_mean_mean', 0),
                'avg_forgetting': forgetting.get('avg_forgetting_rate', 0)
            })

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Dice comparison
        axes[0].bar(comp_df['experiment'], comp_df['dice_mean'], color='skyblue')
        axes[0].set_title('Dice Coefficient Comparison')
        axes[0].set_ylabel('Dice Score')
        axes[0].tick_params(axis='x', rotation=45)

        # HD95 comparison
        axes[1].bar(comp_df['experiment'], comp_df['hd95_mean'], color='lightcoral')
        axes[1].set_title('HD95 Comparison')
        axes[1].set_ylabel('HD95 (mm)')
        axes[1].tick_params(axis='x', rotation=45)

        # Forgetting comparison
        axes[2].bar(comp_df['experiment'], comp_df['avg_forgetting'], color='lightgreen')
        axes[2].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Target')
        axes[2].set_title('Forgetting Rate Comparison')
        axes[2].set_ylabel('Forgetting Rate')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].legend()

        plt.suptitle('Experiment Comparison', fontsize=16)
        plt.tight_layout()

        # Save to first experiment's plots directory
        plt.savefig(exp_dirs[0] / "plots" / "experiment_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Experiment comparison completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment Visualization")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--plot", nargs='+', choices=['metrics', 'tasks', 'forgetting', 'clients', 'convergence', 'dashboard', 'all'],
                       default=['all'], help="Plots to generate")
    parser.add_argument("--compare", nargs='+', help="Compare multiple experiments (dir1:exp1 dir2:exp2)")

    args = parser.parse_args()

    if args.compare:
        # Parse comparison arguments
        exp_dirs = []
        exp_names = []
        for comp_arg in args.compare:
            exp_dir, exp_name = comp_arg.split(':')
            exp_dirs.append(Path(exp_dir))
            exp_names.append(exp_name)

        compare_experiments(exp_dirs, exp_names)
    else:
        # Single experiment visualization
        visualizer = ExperimentVisualizer(args.results_dir)

        if 'all' in args.plot or 'metrics' in args.plot:
            visualizer.plot_metrics_over_rounds()

        if 'all' in args.plot or 'tasks' in args.plot:
            visualizer.plot_task_performance()

        if 'all' in args.plot or 'forgetting' in args.plot:
            visualizer.plot_forgetting_analysis()

        if 'all' in args.plot or 'clients' in args.plot:
            visualizer.plot_client_performance_distribution()

        if 'all' in args.plot or 'convergence' in args.plot:
            visualizer.plot_convergence_analysis()

        if 'all' in args.plot or 'dashboard' in args.plot:
            visualizer.create_summary_dashboard()

        # Generate text report
        report = visualizer.generate_experiment_report()
        print("\n" + report)
