"""Helpers for stable residue identifiers across chains and insertion codes."""

from __future__ import annotations

import pandas as pd


def _normalize_chain_id(value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text or "_"


def _normalize_insertion_code(value) -> str:
    return "" if pd.isna(value) else str(value).strip()


def residue_uid(chain_id, resid, insertion_code="") -> str:
    return f"{_normalize_chain_id(chain_id)}:{int(resid)}{_normalize_insertion_code(insertion_code)}"


def residue_uid_from_biopdb(res) -> str:
    return residue_uid(res.get_parent().id, res.id[1], res.id[2])


def has_explicit_residue_keys(df: pd.DataFrame) -> bool:
    if "chain_id" not in df.columns and "insertion_code" not in df.columns:
        return False
    chain_nonempty = "chain_id" in df.columns and df["chain_id"].fillna("").astype(str).str.strip().ne("").any()
    icode_nonempty = "insertion_code" in df.columns and df["insertion_code"].fillna("").astype(str).str.strip().ne("").any()
    return bool(chain_nonempty or icode_nonempty)


def ensure_residue_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "chain_id" not in out.columns:
        out["chain_id"] = ""
    if "insertion_code" not in out.columns:
        out["insertion_code"] = ""

    out["chain_id"] = out["chain_id"].fillna("").astype(str)
    out["insertion_code"] = out["insertion_code"].fillna("").astype(str)

    if "res_uid" not in out.columns:
        has_explicit_keys = has_explicit_residue_keys(out)
        if has_explicit_keys:
            out["res_uid"] = [
                residue_uid(chain_id, resid, icode)
                for chain_id, resid, icode in zip(
                    out["chain_id"], out["resid"], out["insertion_code"]
                )
            ]
        else:
            out["res_uid"] = out["resid"].astype(str)
    else:
        out["res_uid"] = out["res_uid"].astype(str)

    return out


def sort_residue_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "res_order" in df.columns:
        return df.sort_values("res_order", kind="stable")

    sort_cols = [
        col for col in ("chain_id", "resid", "insertion_code")
        if col in df.columns
    ]
    if sort_cols and any(col != "resid" for col in sort_cols):
        return df.sort_values(sort_cols, kind="stable")

    return df.copy()
