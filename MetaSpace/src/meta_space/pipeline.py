# pipeline.py
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import click
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from pathlib import Path

Root_project = Path(__file__).resolve().parents[2]

from embed_utils import build_column_texts, embed_texts, pick_model_checkpoint
from golden_tools import _non_index_columns, golden_matrix_s1xs2
from meta_features.meta_features import  compute_distances_feat

def read_csv_any(path: str | Path) -> pd.DataFrame:
    """Read a CSV with a robust delimiter heuristic."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, sep=None, engine="python")


def compute_distances(
    embedd_ds1: pd.DataFrame,
    embedd_ds2: pd.DataFrame,
    golden_matrix: pd.DataFrame,
    category: str,
    relation: str,
    dataset: str,
    model_name: str,
    inter_csv_path: Optional[Path] = None,
    flush_every: int = 1000,
) -> pd.DataFrame:

    return compute_distances_feat(embedd_ds1,
    embedd_ds2,
    golden_matrix,
    category,
    relation,
    dataset,
    model_name,
    flush_every)


@click.command(name="MetaSpace", context_settings={"show_default": True})
@click.option("--dataset", required=True, help="Dataset name for bookkeeping.")
@click.option(
    "--source-csv",
    "source_csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to source CSV (S1).",
)
@click.option(
    "--target-csv",
    "target_csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to target CSV (S2).",
)
@click.option(
    "--golden-json",
    "golden_json",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to golden JSON.",
)
@click.option(
    "--model",
    "model_alias",
    default="all-MiniLM-L6-v2",
    help="Embedding model alias or HF checkpoint.",
)
@click.option(
    "--out-dir",
    "out_dir",
    default=Root_project / "tests/results_meta_space",
    type=click.Path(file_okay=False),
    help="Output directory.",
)
def meta_match(
    dataset: str,
    source_csv: str,
    target_csv: str,
    golden_json: Optional[str],
    model_alias: str,
    out_dir: str,
) -> None:
    """Run MetaSpace: read CSVs, embed columns, build golden matrix, compute features, save results."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_src = read_csv_any(source_csv)
    df_tgt = read_csv_any(target_csv)

    src_attrs = _non_index_columns(df_src)
    tgt_attrs = _non_index_columns(df_tgt)

    model_ckpt = pick_model_checkpoint(model_alias)

    texts_src = build_column_texts(df_src, colnames=src_attrs, n_samples_per_col=20)
    texts_tgt = build_column_texts(df_tgt, colnames=tgt_attrs, n_samples_per_col=20)

    emb_src = embed_texts(texts_src, model_name=model_ckpt, device=None, normalize=True)
    emb_tgt = embed_texts(texts_tgt, model_name=model_ckpt, device=None, normalize=True)

    if golden_json:
        G = golden_matrix_s1xs2(golden_json, src_attrs, tgt_attrs)
    else:
        G = pd.DataFrame()

    inter_csv = (
        out_path / f"inter_{dataset}__{Path(source_csv).stem}__{Path(target_csv).stem}__{Path(model_ckpt).name}.csv"
    )
    df_features = compute_distances(
        embedd_ds1=emb_src,
        embedd_ds2=emb_tgt,
        golden_matrix=G,
        category="Magellan",
        relation="Unionable",
        dataset=dataset,
        model_name=model_alias,
        inter_csv_path=inter_csv,
        flush_every=2000,
    )
    out_csv = out_path / f"Meta_Space__{dataset}__{Path(model_ckpt).name}.csv"
    df_features.to_csv(out_csv, index=False)
    click.echo(f"[OK] Saved: {out_csv}")


if __name__ == "__main__":
    meta_match()
