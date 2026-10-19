"""Replay the selected shared-model refits from checksum-verified compact inputs.

No evaluation-set tuning, package-weight replacement, or research-tree imports.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from FastHydroMap.v2.model import SharedWaterModel

HERE = Path(__file__).resolve().parent
INPUT_KEYS = ("node", "point", "point_mask", "atom", "edge", "fit_y", "mask")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batch_graphs(graphs, device="cpu"):
    out = {key: torch.cat([g[key] for g in graphs], 0).to(device) for key in INPUT_KEYS}
    offset, edges = 0, []
    for graph in graphs:
        edges.append(graph["edge_index"] + offset)
        offset += len(graph["node"])
    out["edge_index"] = torch.cat(edges, 1).to(device)
    return out


def objective(model, batch, target_std):
    prediction = model(batch)
    terms = []
    for j in range(4):
        keep = batch["mask"][:, j]
        if keep.any():
            p, y = prediction[keep, j], batch["fit_y"][keep, j]
            terms.append(nn.functional.smooth_l1_loss(p, y, beta=.5 / float(target_std[0]))
                         if j == 0 else nn.functional.mse_loss(p, y))
    if not terms:
        raise ValueError("Empty training mask")
    return torch.stack(terms).mean()


def load_training(cache, protocol):
    if sha256(cache) != protocol["cache"]["sha256"]:
        raise ValueError("Cache checksum mismatch; use the frozen full800.pt input")
    rows = torch.load(cache, map_location="cpu", weights_only=True)
    ids = [g["pdb_id"] for g in rows]
    if len(ids) != protocol["cache"]["protein_count"] or len(ids) != len(set(ids)):
        raise ValueError("Cache must contain the 800 unique original training proteins")
    declared = json.loads((HERE / "development_split.json").read_text())
    if set(ids) != set(declared) or any(g["role"] != declared[g["pdb_id"]] for g in rows):
        raise ValueError("Development split does not match the checked cache")
    fitting = set(protocol["fit_ids"])
    if fitting != set(declared):
        raise ValueError("Final fit IDs must contain all 800 original training proteins")
    # Preserve cache order: sorting here would change seeded minibatch sampling.
    train = [g for g in rows if g["pdb_id"] in fitting]
    if len(train) != protocol["cache"]["fit_protein_count"]:
        raise ValueError("Incorrect number of fitting proteins")
    for graph in train:
        n = len(graph["node"])
        shapes = {"node": (n, 35), "point": (n, 24, 11), "point_mask": (n, 24),
                  "atom": (n, 24, 8, 19), "fit_y": (n, 4), "mask": (n, 4)}
        if any(tuple(graph[key].shape) != shape for key, shape in shapes.items()):
            raise ValueError(f"Unexpected feature dimensions: {graph['pdb_id']}")
        if graph["mask"].dtype != torch.bool or graph["point_mask"].dtype != torch.bool:
            raise ValueError("Training masks must be boolean")
        for key in ("node", "point", "atom", "edge"):
            if not torch.isfinite(graph[key]).all():
                raise ValueError(f"Nonfinite {key}: {graph['pdb_id']}")
        if not torch.isfinite(graph["fit_y"][graph["mask"]]).all():
            raise ValueError("Nonfinite eligible training targets")
    return train


def train_seed(train, protocol, seed, output, device, smoke_steps=None):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = SharedWaterModel().to(device)
    config = protocol["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    selected = protocol["seeds"][str(seed)]
    steps = smoke_steps if smoke_steps is not None else selected["steps"]
    if steps < 1 or steps > selected["steps"]:
        raise ValueError("Smoke steps must be between 1 and the selected stopping step")
    out = output / f"seed{seed}"
    out.mkdir()
    start = time.monotonic()
    losses, history = [], []
    for step in range(1, steps + 1):
        lr = selected["learning_rate_at_block_end"][str(((step - 1) // 100 + 1) * 100)]
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        batch = batch_graphs([train[i] for i in rng.choice(len(train), config["batch_proteins"], replace=False)], device)
        value = objective(model, batch, protocol["target_normalization"]["std"])
        if not torch.isfinite(value):
            raise ValueError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_norm"], error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(value.detach()))
        if step % 100 == 0 or step == steps:
            history.append(dict(step=step, loss=float(np.mean(losses)), lr=lr,
                                seconds=time.monotonic() - start))
            print(f"seed={seed} step={step}/{steps} loss={history[-1]['loss']:.6f}", flush=True)
            losses = []
    metadata = dict(status="smoke_test_only" if smoke_steps is not None else "reproduced_refit_not_release_weights",
        architecture=protocol["architecture"], feature_version=protocol["feature_version"],
        width=16, seed=seed, targets=protocol["targets"], arm="control", single_target=None,
        training_stage="full800",
        representation="post_message_passing_pre_prediction_heads", fit_proteins=len(train),
        fit_ids=protocol["fit_ids"], feature_normalization=protocol["feature_normalization"],
        target_normalization=protocol["target_normalization"], best_step=steps, device=device,
        protocol_sha256=sha256(output / "shared_protocol.json"), cache_sha256=protocol["cache"]["sha256"],
        training_script_sha256=sha256(__file__), torch_version=str(torch.__version__))
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, metadata=metadata), out / "checkpoint.pt")
    (out / "SUMMARY.json").write_text(json.dumps(dict(metadata, checkpoint_sha256=sha256(out / "checkpoint.pt")), indent=2) + "\n")
    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43], choices=[17, 29, 43])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    if args.threads < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Use positive thread counts and unique seeds")
    if not args.check_only and (args.output is None or args.output.exists()):
        parser.error("Provide a new --output directory; existing directories are never overwritten")
    protocol_text = (HERE / "shared_protocol.json").read_text()
    protocol = json.loads(protocol_text)
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= min(protocol["seeds"][str(s)]["steps"] for s in args.seeds):
        parser.error("Smoke steps must be positive and no greater than the selected stopping step")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    train = load_training(args.cache, protocol)
    print(f"Verified cache; {len(train)} fitting proteins. Original validation/test cohorts excluded.", flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True)
    (args.output / "shared_protocol.json").write_text(protocol_text)
    for seed in args.seeds:
        train_seed(train, protocol, seed, args.output, args.device, args.smoke_steps)


if __name__ == "__main__":
    main()
