import jax
import jax.numpy as jnp
import flax.nnx as nnx
from orbax import checkpoint
import orbax
from pathlib import Path
import pandas as pd
import numpy as np
from jax.sharding import SingleDeviceSharding
from rouge_score import rouge_scorer
from tqdm import tqdm
import wandb
import yaml

from data_loader import tokenizer
from model import MiniGPT
from inference import detect_red_team_prompt   # reuse your fixed inference fn

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


# ── ROUGE Evaluation ──────────────────────────────────────────────────────────

def extract_label(decoded_output: str) -> str:
    """
    Pull out only the label portion from the full model output.

    The model generates:
        <|startofprompt|>...<|endofprompt|><|startoflabel|>LABEL<|endoflabel|>

    We only want to score LABEL against the ground truth.

    Args:
        decoded_output : full string returned by generate_text()

    Returns:
        label string, or empty string if delimiters not found
    """
    try:
        start = decoded_output.index("<|startoflabel|>") + len("<|startoflabel|>")
        # end   = decoded_output.index("<|endoflabel|>", start)
        return decoded_output[start:].strip()
    except ValueError:
        # model didn't produce a complete label — treat as empty prediction
        return ""


def compute_rouge(
    predictions: list[str],
    references:  list[str],
) -> dict[str, dict[str, float]]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L across a list of prediction/reference pairs.

    Args:
        predictions : list of model-predicted label strings
        references  : list of ground-truth label strings

    Returns:
        dict with keys "rouge1", "rouge2", "rougeL", each containing:
            precision, recall, f1  (macro-averaged over all samples)
    """
    scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        scores = scorer.score(target=ref, prediction=pred)
        for key in results:
            results[key].append(scores[key])

    # Macro-average precision / recall / F1 across all samples
    aggregated = {}
    for key, score_list in results.items():
        aggregated[key] = {
            "precision": float(np.mean([s.precision for s in score_list])),
            "recall"   : float(np.mean([s.recall    for s in score_list])),
            "f1"       : float(np.mean([s.fmeasure  for s in score_list])),
        }

    return aggregated


def run_eval(
    model,
    eval_df:        pd.DataFrame,
    temperature:    float = 0.0,   # greedy by default for eval
    max_new_tokens: int   = 100,
    log_to_wandb:   bool  = True,
    save_results:   bool  = True,
    output_path:    str   = "eval_results.csv",
) -> dict[str, dict[str, float]]:
    """
    Run full evaluation on an eval split and report ROUGE scores.

    Args:
        model          : loaded MiniGPT instance
        eval_df        : DataFrame with columns ["text", "category"]
        temperature    : sampling temperature (0.0 = greedy, best for eval)
        max_new_tokens : max tokens to generate per sample
        log_to_wandb   : whether to push scores to W&B
        save_results   : whether to save per-sample CSV
        output_path    : path for the per-sample results CSV

    Returns:
        ROUGE score dict  {"rouge1": {"precision", "recall", "f1"}, ...}
    """
    predictions = []
    references  = []
    raw_outputs = []

    print(f"Evaluating {len(eval_df)} samples (temperature={temperature}) ...\n")

    for i, (_, row) in enumerate(tqdm(eval_df.iterrows(), total=len(eval_df), desc="Eval")):
        raw_out = detect_red_team_prompt(
            model,
            raw_prompt=row["text"],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=i,
        )
        pred_label = extract_label(raw_out)

        predictions.append(pred_label)
        references.append(str(row["category"]).strip())
        raw_outputs.append(raw_out)

    # ── ROUGE scores ──────────────────────────────────────────────────────────
    rouge_scores = compute_rouge(predictions, references)

    # ── Pretty print ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    for metric, vals in rouge_scores.items():
        print(
            f"{metric:<12} "
            f"{vals['precision']:>10.4f} "
            f"{vals['recall']:>10.4f} "
            f"{vals['f1']:>10.4f}"
        )
    print("=" * 55 + "\n")

    # ── W&B logging ───────────────────────────────────────────────────────────
    if log_to_wandb:
        flat_scores = {
            f"eval/{metric}/{k}": v
            for metric, vals in rouge_scores.items()
            for k, v in vals.items()
        }
        wandb.log(flat_scores)

        # Log per-sample predictions as a W&B Table for easy inspection
        table = wandb.Table(columns=["prompt", "ground_truth", "prediction", "raw_output"])
        for row_data, pred, ref, raw in zip(
            eval_df.itertuples(), predictions, references, raw_outputs
        ):
            table.add_data(row_data.text, ref, pred, raw)
        wandb.log({"eval/predictions": table})

    # ── Save per-sample CSV ───────────────────────────────────────────────────
    if save_results:
        results_df = eval_df.copy().reset_index(drop=True)
        results_df["prediction"] = predictions
        results_df["raw_output"] = raw_outputs
        results_df.to_csv(output_path, index=False)
        print(f"Per-sample results saved → {output_path}")

    return rouge_scores


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Load model ────────────────────────────────────────────────────────────
    model = MiniGPT(
        vocab_size=tokenizer.n_vocab,
        maxlen=config["MAX_LENGTH"],
        embed_dim=config["EMBED_DIM"],
        num_heads=config["NUM_HEADS"],
        feed_forward_dim=config["FEED_FORWARD_DIM"],
        num_transformer_blocks=config["NUM_LAYERS"],
        rngs=nnx.Rngs(0),
    )

    cpu_device   = jax.devices("cpu")[0]
    cpu_sharding = SingleDeviceSharding(cpu_device)
    restore_args = jax.tree_util.tree_map(
        lambda _: checkpoint.ArrayRestoreArgs(sharding=cpu_sharding),
        nnx.state(model),
    )

    checkpoint_path = Path.cwd() / "new_model_checkpoint.orbax"
    checkpointer    = orbax.checkpoint.PyTreeCheckpointer()
    restored_state  = checkpointer.restore(
        checkpoint_path,
        item=nnx.state(model),
        restore_args=restore_args,
    )
    nnx.update(model, restored_state)
    print("Checkpoint loaded ✅")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project="minigpt-pretraining",
        job_type="eval",
        config={
            "temperature"   : 0.0,
            "max_new_tokens": 100,
            "eval_file"     : "eval_data.csv",
        },
    )

    # ── Load eval data ────────────────────────────────────────────────────────
    eval_df = pd.read_csv("test_data.csv")

    # ── Run eval ──────────────────────────────────────────────────────────────
    scores = run_eval(
        model,
        eval_df,
        temperature=0.0,      # greedy for deterministic eval
        max_new_tokens=100,
        log_to_wandb=True,
        save_results=True,
        output_path="eval_results.csv",
    )

    wandb.finish()