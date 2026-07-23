"""Run all DDPM experiments in configs_ddpm.EXPERIMENTS, append a markdown
summary to RESULTS.md, and generate a sample figure for each.

Latent runs assume the referenced VAE checkpoint (config["vae_name"]) already
exists in checkpoints/ — train the VAE first with the existing run_experiments.py.

Usage: uv run python run_experiments_ddpm.py
"""

import os
from datetime import datetime

from configs_ddpm import EXPERIMENTS
from inference_ddpm import visualize
from train_ddpm import CHECKPOINT_DIR, run_experiment


RESULTS_PATH = "RESULTS.md"


def format_results_section(results):
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"\n## DDPM Experiment Sweep — {today}\n")
    lines.append("DDPM on MNIST, epsilon-prediction with MSE, Adam. Pixel runs use a ")
    lines.append("UNet on images in [-1, 1]; latent runs diffuse frozen-VAE latents ")
    lines.append("with an MLP denoiser and decode through the VAE.\n\n")

    lines.append("### Summary table\n\n")
    lines.append("| Run | space | timesteps | epochs | Final train loss | Final test loss |\n")
    lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for r in results:
        c = r["config"]
        lines.append(
            f"| `{r['name']}` | {c['space']} | {c['num_timesteps']} | {c['num_epochs']} | "
            f"{r['final_train_loss']:.4f} | **{r['final_test_loss']:.4f}** |\n"
        )

    lines.append("\n### Per-run details\n\n")
    for r in results:
        c = r["config"]
        lines.append(f"- **`{r['name']}`** — space={c['space']}, timesteps={c['num_timesteps']}, ")
        lines.append(f"epochs={c['num_epochs']}, final test loss **{r['final_test_loss']:.4f}**. ")
        lines.append(f"Figure: `figures/{r['name']}_samples.png`.\n")

    return "".join(lines)


def main():
    results = []
    for config in EXPERIMENTS:
        name = config["name"]
        print(f"\n=== Running {name} ===")
        result = run_experiment(config)
        result["config"] = config
        results.append(result)

        print(f"--- inference for {name} ---")
        visualize(name)

    section = format_results_section(results)
    with open(RESULTS_PATH, "a") as f:
        f.write(section)
    print(f"\nAppended DDPM sweep results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
