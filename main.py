"""
Relational Cross-Attention for ARC-like Transform Learning

This experiment demonstrates that multi-layer cross-attention between input and output
grids enables better generalization on unseen geometric and color transformations.

Architecture features (Relational Model):
- Cross-attention pathway: Learns spatial mappings (rotate, flip, translate)
- Dual prediction heads with learned gating:
  * Spatial head: Uses attended embeddings (for geometric transforms)
  * Color head: Uses attended embeddings + one-hot colors (for color transforms)
  * Gate network: Learns which head to trust for each position
- Prevents task interference between spatial and color learning

Baseline (Standard Model):
- Standard transformer with self-attention over all examples + test
- No explicit relational structure
"""

import torch
from src.utils.gpu import detect_gpu
from src.experiments.runner import run_experiment, run_arc_experiment


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the relational cross-attention experiment."""

    # Detect GPU
    device, is_rocm = detect_gpu()

    if is_rocm:
        # Enable optimizations for AMD GPUs
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print("[OK] ROCm optimizations enabled\n")

    print("=" * 70)
    print("Relational Cross-Attention Experiment")
    print("=" * 70)
    print()
    print("Choose experiment:")
    print("  1. Synthetic ARC-lite (default)")
    print("  2. Real ARC dataset")
    print()

    choice = input("Enter choice [1]: ").strip() or "1"
    print()

    if choice == "2":
        data_dir = input("Enter path to ARC-AGI/data folder: ").strip()
        epochs = input("Enter number of epochs [50]: ").strip() or "50"
        run_arc_experiment(data_dir, epochs=int(epochs), device=device)
        return

    # Experiment configuration - testing both spatial AND color transforms
    train_transforms = ['rotate_90', 'flip_horizontal', 'translate_right', 'increment_colors']
    test_transforms = ['rotate_180', 'flip_vertical', 'translate_down', 'invert_colors']
    seeds = [7200, 4995]
    epochs = 30

    print(f"TRAIN transforms: {train_transforms}")
    print(f"TEST transforms:  {test_transforms}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {epochs}")
    print(f"Model: embed_dim=128, num_layers=4, num_heads=4")
    print()

    # Store results for both models
    all_results = {
        'standard': [],
        'relational': []
    }

    # Run experiment for each seed and model type
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}")

        for model_type in ['standard', 'relational']:
            print(f"\n[{model_type.upper()}] Training...")

            result = run_experiment(
                seed=seed,
                train_transforms=train_transforms,
                test_transforms=test_transforms,
                model_type=model_type,
                epochs=epochs,
                device=device
            )

            all_results[model_type].append(result)

            print(f"  Seen: {100*result['final_seen_accuracy']:.2f}% | "
                  f"Unseen: {100*result['final_unseen_accuracy']:.2f}% | "
                  f"Gap: {100*(result['final_seen_accuracy'] - result['final_unseen_accuracy']):.2f}% | "
                  f"Time: {result['total_time']:.1f}s")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print()

    for model_type in ['standard', 'relational']:
        print(f"\n{model_type.upper()} MODEL:")
        print(f"{'Seed':<8} {'Seen Acc':<12} {'Unseen Acc':<12} {'Gap':<10} {'Time':<10}")
        print("-" * 70)

        seen_accs = []
        unseen_accs = []
        gaps = []
        times = []

        for result in all_results[model_type]:
            seen = result['final_seen_accuracy']
            unseen = result['final_unseen_accuracy']
            gap = seen - unseen
            time_taken = result['total_time']

            seen_accs.append(seen)
            unseen_accs.append(unseen)
            gaps.append(gap)
            times.append(time_taken)

            print(f"{result['seed']:<8} {100*seen:<12.2f} {100*unseen:<12.2f} "
                  f"{100*gap:<10.2f} {time_taken:<10.1f}s")

        # Averages
        avg_seen = sum(seen_accs) / len(seen_accs)
        avg_unseen = sum(unseen_accs) / len(unseen_accs)
        avg_gap = sum(gaps) / len(gaps)
        avg_time = sum(times) / len(times)

        print("-" * 70)
        print(f"{'AVERAGE':<8} {100*avg_seen:<12.2f} {100*avg_unseen:<12.2f} "
              f"{100*avg_gap:<10.2f} {avg_time:<10.1f}s")

        # Standard deviations
        std_seen = (sum((x - avg_seen)**2 for x in seen_accs) / len(seen_accs))**0.5
        std_unseen = (sum((x - avg_unseen)**2 for x in unseen_accs) / len(unseen_accs))**0.5

        print(f"Std Dev:  ±{100*std_seen:<11.2f} ±{100*std_unseen:<11.2f}")

    # Per-transform breakdown comparison
    print("\n" + "=" * 70)
    print("PER-TRANSFORM COMPARISON (Unseen, Last Seed)")
    print("=" * 70)
    print()

    std_last = all_results['standard'][-1]
    rel_last = all_results['relational'][-1]


    print(f"{'Transform':<20} {'Standard':<12} {'Relational':<12} {'Delta':<10}")
    print("-" * 70)

    for transform in sorted(std_last['final_unseen_per_transform'].keys()):
        std_acc = std_last['final_unseen_per_transform'][transform]
        rel_acc = rel_last['final_unseen_per_transform'][transform]
        delta = rel_acc - std_acc

        print(f"{transform:<20} {100*std_acc:<12.2f} {100*rel_acc:<12.2f} "
              f"{'+' if delta > 0 else ''}{100*delta:<9.2f}")

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
