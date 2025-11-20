#!/usr/bin/env python
"""
Plot results from all Paper 05 (Omnigrok) experiments
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_json(filepath):
    """Load training history from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {filepath}")
        return None

def plot_mnist_results():
    """Plot MNIST grokking results"""
    data = load_json("results/logs/training_history.json")
    if not data:
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = np.array(data['epoch'])
    train_acc = np.array(data['train_acc']) * 100
    test_acc = np.array(data['test_acc']) * 100
    
    # Accuracy curves
    axes[0, 0].plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, test_acc, 'r-', label='Test', linewidth=2)
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('MNIST: Accuracy vs Training Steps')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curves
    axes[0, 1].semilogy(epochs, data['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].semilogy(epochs, data['test_loss'], 'r-', label='Test', linewidth=2)
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('MNIST: Loss vs Training Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Generalization gap
    gap = train_acc - test_acc
    axes[1, 0].plot(epochs, gap, 'purple', linewidth=2)
    axes[1, 0].fill_between(epochs, 0, gap, alpha=0.3, color='purple')
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Generalization Gap (%)')
    axes[1, 0].set_title('MNIST: Train - Test Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning trajectory
    sc = axes[1, 1].scatter(train_acc, test_acc, c=epochs, cmap='viridis', s=20, alpha=0.6)
    axes[1, 1].plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect generalization')
    axes[1, 1].set_xlabel('Train Accuracy (%)')
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_title('MNIST: Learning Trajectory')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(sc, ax=axes[1, 1], label='Training Step')
    
    plt.tight_layout()
    plt.savefig('results/mnist_corrected_grokking.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/mnist_corrected_grokking.png")
    plt.close()
    return True

def plot_teacher_student_results():
    """Plot teacher-student results"""
    data = load_json("results/logs/teacher_student_training_history.json")
    if not data:
        return False
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = np.array(data['epoch'])
    train_acc = np.array(data['train_acc']) * 100
    test_acc = np.array(data['test_acc']) * 100
    
    # Accuracy curves
    axes[0, 0].plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, test_acc, 'r-', label='Test', linewidth=2)
    axes[0, 0].set_xlabel('Training Epochs')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Teacher-Student: Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss curves
    axes[0, 1].semilogy(epochs, data['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 1].semilogy(epochs, data['test_loss'], 'r-', label='Test', linewidth=2)
    axes[0, 1].set_xlabel('Training Epochs')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('Teacher-Student: Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # L2 norm
    axes[1, 0].plot(epochs, data['l2_norm'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Training Epochs')
    axes[1, 0].set_ylabel('L2 Norm')
    axes[1, 0].set_title('Teacher-Student: Weight Norm')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Generalization gap
    gap = train_acc - test_acc
    axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
    axes[1, 1].fill_between(epochs, 0, gap, alpha=0.3, color='purple')
    axes[1, 1].set_xlabel('Training Epochs')
    axes[1, 1].set_ylabel('Generalization Gap (%)')
    axes[1, 1].set_title('Teacher-Student: Train - Test Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/teacher_student_grokking.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/teacher_student_grokking.png")
    plt.close()
    return True

def plot_qm9_results():
    """Plot QM9 results"""
    data = load_json("results/logs/qm9_training_history.json")
    if not data:
        return False
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = np.array(data['epoch'])
    train_loss = np.array(data['train_loss'])
    test_loss = np.array(data['test_loss'])
    
    # Loss curves (linear scale)
    axes[0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, test_loss, 'r-', label='Test', linewidth=2)
    axes[0].set_xlabel('Training Epochs')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('QM9: Loss vs Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss curves (log scale)
    axes[1].semilogy(epochs, train_loss, 'b-', label='Train', linewidth=2)
    axes[1].semilogy(epochs, test_loss, 'r-', label='Test', linewidth=2)
    axes[1].set_xlabel('Training Epochs')
    axes[1].set_ylabel('MSE Loss (log scale)')
    axes[1].set_title('QM9: Loss vs Epochs (log)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/qm9_grokking.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/qm9_grokking.png")
    plt.close()
    return True

def main():
    """Plot all available results"""
    print("=" * 60)
    print("Paper 05: Omnigrok - Results Plotting")
    print("=" * 60)
    print()
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    success_count = 0
    
    print("Plotting MNIST results...")
    if plot_mnist_results():
        success_count += 1
    
    print("Plotting Teacher-Student results...")
    if plot_teacher_student_results():
        success_count += 1
    
    print("Plotting QM9 results...")
    if plot_qm9_results():
        success_count += 1
    
    print()
    print("=" * 60)
    print(f"Successfully plotted {success_count}/3 experiments")
    print("=" * 60)
    
    if success_count < 3:
        print("\nNote: Some experiments may still be running.")
        print("Check job status with: squeue -u $(whoami) | grep paper05")

if __name__ == "__main__":
    main()

