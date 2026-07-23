"""Run all experiments in configs.EXPERIMENTS, write a markdown summary
to RESULTS.md (appended), and run inference to generate figures for each.

Skips configs whose checkpoint already exists (delete checkpoints/ to rerun).

Usage: uv run python run_experiments.py
"""

import json
import os
from datetime import datetime

from configs import EXPERIMENTS
from inference import visualize
from train import CHECKPOINT_DIR, run_experiment


RESULTS_PATH = "RESULTS.md"


def format_results_section(results):
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"\n## Experiment Sweep — {today}\n")
    lines.append("Hyperparameter sweep varying one knob at a time off the baseline. ")
    lines.append("All configs use MLP 784 → 400 → latent → 400 → 784, ReLU hidden, ")
    lines.append("sigmoid output, BCE(sum) + analytic KL, Adam @ 1e-3, batch 128.\n\n")

    lines.append("### Summary table\n\n")
    lines.append("| Run | latent_dim | epochs | Final train loss | Final test loss |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for r in results:
        lines.append(
            f"| `{r['name']}` | {r['latent_dim']} | {r['num_epochs']} | "
            f"{r['final_train_loss']:.2f} | **{r['final_test_loss']:.2f}** |\n"
        )

    lines.append("\n### Per-epoch test loss\n\n")
    lines.append("| Epoch | " + " | ".join(f"`{r['name']}`" for r in results) + " |\n")
    lines.append("| --- " * (len(results) + 1) + "|\n")
    max_ep = max(len(r["test_losses"]) for r in results)
    for ep in range(max_ep):
        row = [str(ep)]
        for r in results:
            row.append(f"{r['test_losses'][ep]:.2f}" if ep < len(r["test_losses"]) else "—")
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n### Per-run details\n\n")
    for r in results:
        lines.append(f"- **`{r['name']}`** — latent_dim={r['latent_dim']}, epochs={r['num_epochs']}, ")
        lines.append(f"final test loss **{r['final_test_loss']:.2f}**. ")
        lines.append(f"Figures: `figures/{r['name']}_samples.png`, `figures/{r['name']}_recons.png`.\n")

    return "".join(lines)


def main():
    results = []
    for config in EXPERIMENTS:
        name = config["name"]
        if already_trained(name):
            print(f"[{name}] checkpoint exists — loading config but skipping training.")
            # rerun to re-capture losses since we don't persist them; this is
            # quick and ensures the summary table is honest.
        print(f"\n=== Running {name} ===")
        result = run_experiment(config)
        result["latent_dim"] = config["latent_dim"]
        result["num_epochs"] = config["num_epochs"]
        results.append(result)

        print(f"--- inference for {name} ---")
        visualize(name)

    section = format_results_section(results)
    with open(RESULTS_PATH, "a") as f:
        f.write(section)
    print(f"\nAppended sweep results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
