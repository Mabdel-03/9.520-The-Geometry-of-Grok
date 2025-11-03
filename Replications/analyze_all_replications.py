#!/usr/bin/env python3
"""
Analyze and plot train/test loss for all grokking paper replications
Extracts metrics from logs and checkpoints, generates comparison plots
"""

import os
import re
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2


class ReplicationAnalyzer:
    """Analyzes results from grokking paper replications"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.papers = {
            "01": "Power et al. (2022) - OpenAI Grok",
            "02": "Liu et al. (2022) - Effective Theory",
            "03": "Nanda et al. (2023) - Progress Measures",
            "04": "Wang et al. (2024) - Knowledge Graphs",
            "05": "Liu et al. (2022) - Omnigrok",
            "06": "Humayun et al. (2024) - Deep Networks",
            "07": "Thilak et al. (2022) - Slingshot",
            "08": "Doshi et al. (2024) - Modular Polynomials",
            "09": "Levi et al. (2023) - Linear Estimators",
            "10": "Minegishi et al. (2023) - Lottery Tickets",
        }
        self.results = {}
        
    def extract_from_logs(self, log_files: List[Path]) -> Dict:
        """Extract train/test metrics from log files"""
        metrics = {"epoch": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        # Pattern 1: "Epoch XXX | Train: loss/acc | Test: loss/acc"
                        match = re.search(r'Epoch\s+(\d+).*Train.*?([0-9.]+).*?([0-9.]+).*Test.*?([0-9.]+).*?([0-9.]+)', line)
                        if match:
                            epoch, trl, tra, tel, tea = match.groups()
                            metrics["epoch"].append(int(epoch))
                            metrics["train_loss"].append(float(trl))
                            metrics["train_acc"].append(float(tra))
                            metrics["test_loss"].append(float(tel))
                            metrics["test_acc"].append(float(tea))
                            continue
                        
                        # Pattern 2: Separate train and test lines
                        train_match = re.search(r'train.*loss[:\s]+([0-9.]+).*acc[:\s]+([0-9.]+)', line, re.IGNORECASE)
                        test_match = re.search(r'test.*loss[:\s]+([0-9.]+).*acc[:\s]+([0-9.]+)', line, re.IGNORECASE)
                        epoch_match = re.search(r'epoch[:\s]+(\d+)', line, re.IGNORECASE)
                        
                        if train_match or test_match or epoch_match:
                            # This is a simpler pattern - may need refinement
                            pass
                            
            except Exception as e:
                print(f"Warning: Could not read {log_file}: {e}")
                
        return metrics
    
    def extract_from_json(self, json_files: List[Path]) -> Dict:
        """Extract metrics from JSON checkpoint files"""
        metrics = {"epoch": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Try different possible keys
                        for key in ['epoch', 'step', 'iteration']:
                            if key in data:
                                metrics["epoch"].append(data[key])
                                break
                        for key in ['train_loss', 'loss']:
                            if key in data:
                                metrics["train_loss"].append(data[key])
                                break
                        for key in ['test_loss', 'val_loss']:
                            if key in data:
                                metrics["test_loss"].append(data[key])
                                break
                        for key in ['train_acc', 'train_accuracy']:
                            if key in data:
                                metrics["train_acc"].append(data[key])
                                break
                        for key in ['test_acc', 'val_acc', 'test_accuracy']:
                            if key in data:
                                metrics["test_acc"].append(data[key])
                                break
            except Exception as e:
                print(f"Warning: Could not read {json_file}: {e}")
                
        return metrics
    
    def analyze_paper(self, paper_num: str) -> Optional[Dict]:
        """Analyze results for a specific paper"""
        paper_dir = self.base_dir / f"{paper_num}_*"
        matching_dirs = list(self.base_dir.glob(f"{paper_num}_*"))
        
        if not matching_dirs:
            print(f"Warning: No directory found for paper {paper_num}")
            return None
            
        paper_dir = matching_dirs[0]
        print(f"Analyzing {paper_dir.name}...")
        
        # Look for log files
        log_files = list(paper_dir.glob("logs/*.out")) + list(paper_dir.glob("logs/*.log"))
        metrics = self.extract_from_logs(log_files)
        
        # If no metrics from logs, try JSON files
        if not metrics["epoch"]:
            json_files = list(paper_dir.glob("checkpoints/*.json")) + \
                        list(paper_dir.glob("logs/*.json")) + \
                        list(paper_dir.glob("*.json"))
            metrics_json = self.extract_from_json(json_files)
            if metrics_json["epoch"]:
                metrics = metrics_json
        
        if not metrics["epoch"]:
            print(f"  Warning: No metrics found for paper {paper_num}")
            return None
            
        print(f"  Found {len(metrics['epoch'])} data points")
        return metrics
    
    def plot_single_paper(self, paper_num: str, metrics: Dict, save_dir: Path):
        """Create plots for a single paper"""
        if not metrics or not metrics["epoch"]:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        epochs = np.array(metrics["epoch"])
        
        # Plot 1: Loss curves
        ax = axes[0]
        if metrics["train_loss"]:
            ax.plot(epochs, metrics["train_loss"], label='Train Loss', color='blue', linewidth=2)
        if metrics["test_loss"]:
            ax.plot(epochs, metrics["test_loss"], label='Test Loss', color='red', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{self.papers[paper_num]}\nLoss vs Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Plot 2: Accuracy curves
        ax = axes[1]
        if metrics["train_acc"]:
            ax.plot(epochs, metrics["train_acc"], label='Train Accuracy', color='blue', linewidth=2)
        if metrics["test_acc"]:
            ax.plot(epochs, metrics["test_acc"], label='Test Accuracy', color='red', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{self.papers[paper_num]}\nAccuracy vs Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        save_path = save_dir / f"paper_{paper_num}_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {save_path}")
    
    def plot_comparison(self, save_dir: Path):
        """Create comparison plots across all papers"""
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        axes = axes.flatten()
        
        for idx, (paper_num, metrics) in enumerate(sorted(self.results.items())):
            if metrics and metrics["epoch"]:
                ax = axes[idx]
                epochs = np.array(metrics["epoch"])
                
                # Normalize epoch range to [0, 1] for comparison
                if len(epochs) > 1:
                    norm_epochs = (epochs - epochs[0]) / (epochs[-1] - epochs[0])
                else:
                    norm_epochs = epochs
                
                if metrics["train_loss"]:
                    ax.plot(norm_epochs, metrics["train_loss"], label='Train', color='blue', alpha=0.7)
                if metrics["test_loss"]:
                    ax.plot(norm_epochs, metrics["test_loss"], label='Test', color='red', alpha=0.7)
                    
                ax.set_title(f"Paper {paper_num}", fontsize=10)
                ax.set_xlabel('Normalized Progress', fontsize=8)
                ax.set_ylabel('Loss', fontsize=8)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
            else:
                axes[idx].text(0.5, 0.5, f'Paper {paper_num}\nNo Data', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
        
        plt.suptitle('Grokking Across All Papers: Loss Comparison', fontsize=16, y=0.995)
        plt.tight_layout()
        save_path = save_dir / "all_papers_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nSaved comparison plot: {save_path}")
    
    def generate_summary_report(self, save_dir: Path):
        """Generate a text summary of all results"""
        report_path = save_dir / "analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Grokking Paper Replications - Analysis Summary\n")
            f.write("=" * 80 + "\n\n")
            
            for paper_num in sorted(self.papers.keys()):
                f.write(f"\nPaper {paper_num}: {self.papers[paper_num]}\n")
                f.write("-" * 80 + "\n")
                
                if paper_num not in self.results or not self.results[paper_num]:
                    f.write("  Status: No data found\n")
                    continue
                    
                metrics = self.results[paper_num]
                n_epochs = len(metrics["epoch"])
                f.write(f"  Status: Data collected\n")
                f.write(f"  Epochs: {n_epochs}\n")
                
                if metrics["train_loss"]:
                    final_train = metrics["train_loss"][-1]
                    f.write(f"  Final Train Loss: {final_train:.6f}\n")
                if metrics["test_loss"]:
                    final_test = metrics["test_loss"][-1]
                    f.write(f"  Final Test Loss: {final_test:.6f}\n")
                if metrics["train_acc"]:
                    final_train_acc = metrics["train_acc"][-1]
                    f.write(f"  Final Train Accuracy: {final_train_acc:.4f}\n")
                if metrics["test_acc"]:
                    final_test_acc = metrics["test_acc"][-1]
                    f.write(f"  Final Test Accuracy: {final_test_acc:.4f}\n")
                    
                    # Check for grokking (high train acc, initially low test acc that improves)
                    if final_train_acc > 0.95 and len(metrics["test_acc"]) > 10:
                        early_test = np.mean(metrics["test_acc"][:len(metrics["test_acc"])//4])
                        if early_test < 0.5 and final_test_acc > early_test + 0.2:
                            f.write(f"  ** GROKKING DETECTED ** (test acc: {early_test:.2f} -> {final_test_acc:.2f})\n")
                
        print(f"\nSaved summary report: {report_path}")
    
    def run_analysis(self):
        """Run full analysis pipeline"""
        print("\n" + "=" * 80)
        print("Analyzing All Grokking Paper Replications")
        print("=" * 80 + "\n")
        
        # Create output directory
        output_dir = self.base_dir / "analysis_results"
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir}\n")
        
        # Analyze each paper
        for paper_num in sorted(self.papers.keys()):
            metrics = self.analyze_paper(paper_num)
            if metrics:
                self.results[paper_num] = metrics
                self.plot_single_paper(paper_num, metrics, output_dir)
        
        # Generate comparison plots and summary
        if self.results:
            self.plot_comparison(output_dir)
            self.generate_summary_report(output_dir)
        
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)
        print(f"\nResults saved to: {output_dir}")
        print(f"  - Individual plots: paper_XX_metrics.png")
        print(f"  - Comparison plot: all_papers_comparison.png")
        print(f"  - Summary report: analysis_summary.txt")
        print("")


def main():
    parser = argparse.ArgumentParser(description="Analyze grokking paper replication results")
    parser.add_argument("--dir", type=str, default=".", 
                       help="Base directory containing replication folders")
    args = parser.parse_args()
    
    analyzer = ReplicationAnalyzer(args.dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

