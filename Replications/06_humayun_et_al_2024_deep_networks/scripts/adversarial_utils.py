"""
Adversarial attack utilities for Deep Networks Always Grok
Implements PGD (Projected Gradient Descent) attacks for testing delayed robustness
"""

import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon, alpha=None, num_iter=10, device='cuda'):
    """
    Perform L-infinity PGD attack on images
    
    Args:
        model: Neural network model
        images: Input images (batch)
        labels: True labels
        epsilon: L-infinity perturbation bound (e.g., 0.06, 0.10, 0.13, 0.16, 0.20)
        alpha: Step size (default: epsilon/4)
        num_iter: Number of PGD iterations (default: 10)
        device: Device to run on
    
    Returns:
        adv_images: Adversarially perturbed images
    """
    if alpha is None:
        alpha = epsilon / 4
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Clone images and enable gradient tracking
    adv_images = images.clone().detach().to(device)
    adv_images.requires_grad = True
    
    # Store original images for projection
    original_images = images.clone().detach().to(device)
    
    # Get original data bounds for clamping
    data_min = original_images.min().item()
    data_max = original_images.max().item()
    
    for _ in range(num_iter):
        # Ensure gradients are enabled for this iteration
        adv_images = adv_images.detach()
        adv_images.requires_grad = True
        
        # Forward pass with gradient enabled (use torch.enable_grad to force gradient computation)
        with torch.enable_grad():
            outputs = model(adv_images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
        
        # Update adversarial images
        with torch.no_grad():
            # Sign of gradient
            grad_sign = adv_images.grad.sign()
            
            # Take step in direction of gradient sign
            adv_images = adv_images + alpha * grad_sign
            
            # Project back to epsilon ball around original images
            perturbation = torch.clamp(adv_images - original_images, -epsilon, epsilon)
            adv_images = original_images + perturbation
            
            # Clamp to valid data range
            adv_images = torch.clamp(adv_images, data_min, data_max)
    
    return adv_images.detach()


def evaluate_adversarial_accuracy(model, data_loader, epsilon, device='cuda', 
                                  alpha=None, num_iter=10, max_batches=None):
    """
    Evaluate model accuracy on adversarial examples
    
    Args:
        model: Neural network model
        data_loader: DataLoader for test data
        epsilon: L-infinity perturbation bound
        device: Device to run on
        alpha: Step size for PGD (default: epsilon/4)
        num_iter: Number of PGD iterations
        max_batches: Maximum number of batches to evaluate (None = all)
    
    Returns:
        accuracy: Adversarial accuracy (0-1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            images, labels = images.to(device), labels.to(device)
            
            # Generate adversarial examples
            adv_images = pgd_attack(model, images, labels, epsilon, 
                                   alpha=alpha, num_iter=num_iter, device=device)
            
            # Evaluate on adversarial examples
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def evaluate_multiple_epsilons(model, data_loader, epsilons, device='cuda',
                               alpha=None, num_iter=10, max_batches=None):
    """
    Evaluate model accuracy on adversarial examples with multiple epsilon values
    
    Args:
        model: Neural network model
        data_loader: DataLoader for test data
        epsilons: List of epsilon values to test
        device: Device to run on
        alpha: Step size for PGD (default: epsilon/4 for each)
        num_iter: Number of PGD iterations
        max_batches: Maximum number of batches to evaluate per epsilon
    
    Returns:
        results: Dictionary mapping epsilon to adversarial accuracy
    """
    results = {}
    
    for eps in epsilons:
        acc = evaluate_adversarial_accuracy(
            model, data_loader, eps, device=device,
            alpha=alpha, num_iter=num_iter, max_batches=max_batches
        )
        results[eps] = acc
    
    return results

