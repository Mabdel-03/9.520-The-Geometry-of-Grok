"""
Analyze Paper 4 training results
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("PAPER 4 (WANG ET AL. 2024) - TRAINING RESULTS ANALYSIS")
print("=" * 70)
print()

# Load training progress
df = pd.read_csv('training_progress_scores.csv')

print("1. TRAINING COMPLETION STATUS")
print("-" * 70)
print(f"   Total steps logged: {len(df):,}")
print(f"   Final step: {df['global_step'].max():,}")
print(f"   Target steps: 100,000")
print(f"   Completion: {'✅ COMPLETE' if df['global_step'].max() >= 100000 else '⚠️ Incomplete'}")
print()

print("2. FINAL METRICS")
print("-" * 70)
final = df.iloc[-1]
print(f"   Final step: {final['global_step']:,.0f}")
print(f"   Final epoch: {final['epoch']:,.0f}")
print(f"   Final train loss: {final['train_loss']:.2e}")
print()

# Get evaluation loss checkpoints (where eval_loss != -1)
eval_checkpoints = df[df['eval_loss'] != -1.0]
print(f"   Evaluation checkpoints: {len(eval_checkpoints)}")
print()

print("3. EVALUATION LOSS AT CHECKPOINTS")
print("-" * 70)
for _, row in eval_checkpoints.iterrows():
    step = int(row['global_step'])
    eval_loss = row['eval_loss']
    print(f"   Step {step:>6,}: Eval Loss = {eval_loss:.6e}")
print()

# Analyze grokking transition
print("4. GROKKING TRANSITION ANALYSIS")
print("-" * 70)
if len(eval_checkpoints) > 1:
    for i in range(1, len(eval_checkpoints)):
        prev_step = int(eval_checkpoints.iloc[i-1]['global_step'])
        curr_step = int(eval_checkpoints.iloc[i]['global_step'])
        prev_loss = eval_checkpoints.iloc[i-1]['eval_loss']
        curr_loss = eval_checkpoints.iloc[i]['eval_loss']
        
        # Calculate improvement
        improvement = (prev_loss - curr_loss) / prev_loss * 100 if prev_loss > 0 else 0
        
        if improvement > 50:  # More than 50% improvement
            print(f"   🎯 MAJOR TRANSITION:")
            print(f"      Step {prev_step:,} → {curr_step:,}")
            print(f"      Loss: {prev_loss:.6e} → {curr_loss:.6e}")
            print(f"      Improvement: {improvement:.1f}%")
            print()

print("5. TRAINING LOSS PROGRESSION")
print("-" * 70)
milestones = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 75000, 100000]
for milestone in milestones:
    # Find closest step
    closest_idx = (df['global_step'] - milestone).abs().idxmin()
    row = df.iloc[closest_idx]
    step = int(row['global_step'])
    loss = row['train_loss']
    if loss != -1.0:
        print(f"   Step ~{milestone:>6,}: Train Loss = {loss:.6e}")

print()

print("6. LOSS REDUCTION SUMMARY")
print("-" * 70)
initial_train = df[df['train_loss'] != -1.0].iloc[0]['train_loss']
final_train = df[df['train_loss'] != -1.0].iloc[-1]['train_loss']
reduction = initial_train / final_train

print(f"   Initial train loss: {initial_train:.6e}")
print(f"   Final train loss:   {final_train:.6e}")
print(f"   Reduction factor:   {reduction:.2e}x")
print()

if len(eval_checkpoints) > 1:
    initial_eval = eval_checkpoints.iloc[0]['eval_loss']
    final_eval = eval_checkpoints.iloc[-1]['eval_loss']
    eval_reduction = initial_eval / final_eval if final_eval > 0 else float('inf')
    
    print(f"   Initial eval loss:  {initial_eval:.6e}")
    print(f"   Final eval loss:    {final_eval:.6e}")
    print(f"   Reduction factor:   {eval_reduction:.2e}x")
print()

print("7. PRELIMINARY GROKKING ASSESSMENT")
print("=" * 70)

# Check for grokking indicators
has_large_eval_drop = False
if len(eval_checkpoints) > 1:
    for i in range(1, len(eval_checkpoints)):
        prev_loss = eval_checkpoints.iloc[i-1]['eval_loss']
        curr_loss = eval_checkpoints.iloc[i]['eval_loss']
        improvement = (prev_loss - curr_loss) / prev_loss * 100 if prev_loss > 0 else 0
        if improvement > 50:
            has_large_eval_drop = True
            break

checks = {
    "Training loss near zero": final_train < 1e-5,
    "Evaluation loss near zero": final_eval < 1e-5 if len(eval_checkpoints) > 0 else False,
    "Large eval loss drop detected": has_large_eval_drop,
    "Training completed to 100K steps": df['global_step'].max() >= 100000,
}

for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"   {status} {check}")

print()
passed = sum(checks.values())
total = len(checks)

if passed >= 3:
    print(f"   🎉 STRONG EVIDENCE OF GROKKING ({passed}/{total} checks)")
    print(f"   ✅ Delayed generalization likely occurred")
else:
    print(f"   ⚠️  WEAK EVIDENCE ({passed}/{total} checks)")

print("=" * 70)
print()
print("NOTE: Full verification requires accuracy metrics and test set evaluation")
print()
