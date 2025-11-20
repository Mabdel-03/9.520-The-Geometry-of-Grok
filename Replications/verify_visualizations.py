"""
Verify visualization files exist and create a summary
"""
import os

print("=" * 70)
print("PAPER 3 (NANDA ET AL. 2023) - VISUALIZATION VERIFICATION")
print("=" * 70)
print()

# Check for visualization files
viz_dir = "analysis_results"
result_dir = "03_nanda_et_al_2023_progress_measures/results"

print("1. VISUALIZATION FILES")
print("-" * 70)

# Check analysis_results directory
viz_files = [
    "paper_03_results.png",
    "paper_03_grokking_detailed.png"
]

for file in viz_files:
    path = os.path.join(viz_dir, file)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"   ✅ {file}")
        print(f"      Location: {viz_dir}/")
        print(f"      Size: {size:,} bytes ({size/1024:.1f} KB)")
    else:
        print(f"   ❌ {file} - NOT FOUND")

print()

# Check results directory for additional plots
result_file = os.path.join(result_dir, "paper_03_grokking_detailed.png")
if os.path.exists(result_file):
    size = os.path.getsize(result_file)
    print(f"   ✅ paper_03_grokking_detailed.png")
    print(f"      Location: {result_dir}/")
    print(f"      Size: {size:,} bytes ({size/1024:.1f} KB)")

print()

print("2. EXPECTED VISUALIZATIONS")
print("-" * 70)
print("   These plots should show:")
print("   • Training accuracy curve (reaching 100% quickly)")
print("   • Test accuracy curve (delayed, then jumping)")
print("   • Clear grokking transitions (sudden jumps)")
print("   • Three learning phases visible")
print("   • Generalization gap closing over time")
print()

print("3. VISUALIZATION VERIFICATION SUMMARY")
print("=" * 70)

# Count existing files
existing = sum([os.path.exists(os.path.join(viz_dir, f)) for f in viz_files])
if os.path.exists(result_file):
    existing += 1

if existing >= 2:
    print(f"   ✅ VISUALIZATION FILES PRESENT ({existing} files found)")
    print(f"   ✅ Grokking curves have been generated")
else:
    print(f"   ⚠️  LIMITED VISUALIZATIONS ({existing} files found)")

print("=" * 70)
print()
