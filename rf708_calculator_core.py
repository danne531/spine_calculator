from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ASSET_PATH = Path(__file__).with_name("model_assets.json")


def load_assets(path: str | Path = ASSET_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


ASSETS = load_assets()


ACT_LABEL_MAP_CN = {
    "C.LLB": "颈椎左侧向弯曲",
    "C.RLB": "颈椎右侧向弯曲",
    "T.LLB": "胸椎左侧向弯曲",
    "T.RLB": "胸椎右侧向弯曲",
    "L.LLB": "腰椎左侧向弯曲",
    "L.RLB": "腰椎右侧向弯曲",
    "C.FFT": "颈椎前屈",
    "C.BF": "颈椎后伸",
    "T.FFT": "胸椎前屈",
    "T.BF": "胸椎后伸",
    "L.FFT": "腰椎前屈",
    "L.BF": "腰椎后伸",
    "C.LHR": "颈椎左转",
    "C.RHR": "颈椎右转",
    "T.LHR": "胸椎左转",
    "T.RHR": "胸椎右转",
    "L.LHR": "腰椎左转",
    "L.RHR": "腰椎右转",
}

BAL_LABEL_MAP_CN = {
    "HB": "头部平衡",
    "SB": "肩部平衡",
    "PB": "髋骨平衡",
}

ACT_LABEL_MAP_EN = {
    "C.LLB": "Cervical left lateral bending",
    "C.RLB": "Cervical right lateral bending",
    "T.LLB": "Thoracic left lateral bending",
    "T.RLB": "Thoracic right lateral bending",
    "L.LLB": "Lumbar left lateral bending",
    "L.RLB": "Lumbar right lateral bending",
    "C.FFT": "Cervical flexion",
    "C.BF": "Cervical extension",
    "T.FFT": "Thoracic flexion",
    "T.BF": "Thoracic extension",
    "L.FFT": "Lumbar flexion",
    "L.BF": "Lumbar extension",
    "C.LHR": "Cervical left rotation",
    "C.RHR": "Cervical right rotation",
    "T.LHR": "Thoracic left rotation",
    "T.RHR": "Thoracic right rotation",
    "L.LHR": "Lumbar left rotation",
    "L.RHR": "Lumbar right rotation",
}

BAL_LABEL_MAP_EN = {
    "HB": "Head balance",
    "SB": "Shoulder balance",
    "PB": "Pelvic balance",
}

ST_OPTIONS_CN = {
    "无侧弯": "No",
    "Ⅰ度侧弯": "1degree",
    "Ⅱ度侧弯": "2degree",
    "Ⅲ度侧弯": "3degree",
}

SCT_OPTIONS_CN = {
    "无异常": "No",
    "脊柱前凸（lordosis）": "lordosis",
    "平背（Flat back）": "Flat_back",
}

ST_OPTIONS_EN = {
    "No scoliosis": "No",
    "Grade I scoliosis": "1degree",
    "Grade II scoliosis": "2degree",
    "Grade III scoliosis": "3degree",
}

SCT_OPTIONS_EN = {
    "Normal": "No",
    "Lordosis": "lordosis",
    "Flat back": "Flat_back",
}


def activity_vars() -> list[str]:
    return list(ASSETS["pca_activity_stage1"]["cols"])


def balance_vars() -> list[str]:
    return list(ASSETS["pca_balance_stage1"]["cols"])


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _to_float(value: Any, default: float) -> float:
    if _is_missing(value):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _default_for(var: str) -> Any:
    return ASSETS["defaults"].get(var, "")


def simple_impute_raw(raw_row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    input_types = ASSETS["input_types"]
    for var, default in ASSETS["defaults"].items():
        value = raw_row.get(var, default)
        if input_types.get(var) == "numeric":
            out[var] = _to_float(value, _to_float(default, 0.0))
        else:
            out[var] = default if _is_missing(value) else str(value)
    for key, value in raw_row.items():
        if key not in out:
            out[key] = value
    return out


def zscore_row(raw_row: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw_row)
    means = ASSETS["zscaler"]["mean"]
    sds = ASSETS["zscaler"]["sd"]
    for var in ASSETS["zscaler"]["vars"]:
        mean = float(means[var])
        sd = float(sds[var])
        if abs(sd) < 1e-12:
            sd = 1.0
        out[var] = (_to_float(out.get(var), mean) - mean) / sd
    return out


def predict_svd_pca(row: dict[str, Any], fit: dict[str, Any]) -> dict[str, float]:
    cols = fit["cols"]
    x = np.array([_to_float(row.get(col), 0.0) for col in cols], dtype=float)
    loadings = np.array(fit["loadings"], dtype=float)
    if loadings.ndim == 1:
        loadings = loadings.reshape(-1, 1)

    if fit.get("method") == "external_pca_and_ggplot2":
        if not bool(fit.get("helper_is_log", False)):
            if np.any(x <= -1):
                raise ValueError("PCA projection found values <= -1 and cannot apply log2(x + 1).")
            x_use = np.log2(x + 1.0)
        else:
            x_use = x
        centered = x_use - np.mean(x_use)
        sv_d = np.array(fit["sv_d"], dtype=float)[: loadings.shape[1]]
        sv_d[np.abs(sv_d) <= np.finfo(float).eps] = np.finfo(float).eps
        score = centered @ loadings[:, : len(sv_d)] @ np.diag(1.0 / sv_d)
    else:
        center = np.array(fit["center"], dtype=float)
        score = (x - center) @ loadings

    names = fit["pc_model_names"]
    if isinstance(names, str):
        names = [names]
    return {name: float(score[i]) for i, name in enumerate(names[: len(score)])}


def add_pc1(raw_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    imputed = simple_impute_raw(raw_row)
    z_row = zscore_row(imputed)
    pca_input = imputed if ASSETS.get("use_external_pca_helper") else z_row

    act_scores = predict_svd_pca(pca_input, ASSETS["pca_activity_stage1"])
    bal_scores = predict_svd_pca(pca_input, ASSETS["pca_balance_stage1"])
    stage1 = {**act_scores, **bal_scores}
    integrated = predict_svd_pca(stage1, ASSETS["pca_twostage_integrated"])

    row_with_pc = dict(z_row)
    row_with_pc.update(integrated)
    pc_values = {**stage1, **integrated}
    return row_with_pc, pc_values


def make_model_matrix(row_with_pc: dict[str, Any]) -> dict[str, float]:
    out = {}
    out["TypeCOM"] = 1.0 if str(row_with_pc.get("Type", "")) == "COM" else 0.0
    out["GenderMale"] = 1.0 if str(row_with_pc.get("Gender", "")) == "Male" else 0.0
    out["Age"] = _to_float(row_with_pc.get("Age"), 0.0)
    out["ATI"] = _to_float(row_with_pc.get("ATI"), 0.0)
    out["KA"] = _to_float(row_with_pc.get("KA"), 0.0)
    out["FMR"] = _to_float(row_with_pc.get("FMR"), 0.0)
    st = str(row_with_pc.get("ST", "No"))
    out["ST1degree"] = 1.0 if st == "1degree" else 0.0
    out["ST2degree"] = 1.0 if st == "2degree" else 0.0
    out["ST3degree"] = 1.0 if st == "3degree" else 0.0
    sct = str(row_with_pc.get("SCT", "No"))
    out["SCTFlat_back"] = 1.0 if sct == "Flat_back" else 0.0
    out["SCTlordosis"] = 1.0 if sct == "lordosis" else 0.0
    out["PC1"] = _to_float(row_with_pc.get("PC1"), 0.0)
    return {col: float(out.get(col, 0.0)) for col in ASSETS["model_matrix_columns"]}


def _predict_tree(tree: dict[str, list[Any]], x: dict[str, float]) -> str:
    node = 0
    columns = ASSETS["model_matrix_columns"]
    classes = ASSETS["classes"]
    while True:
        status = int(tree["status"][node])
        if status == -1:
            pred_idx = int(tree["prediction"][node]) - 1
            return classes[pred_idx]
        split_var = int(tree["split_var"][node]) - 1
        split_point = float(tree["split_point"][node])
        value = x[columns[split_var]]
        if value <= split_point:
            node = int(tree["left"][node]) - 1
        else:
            node = int(tree["right"][node]) - 1


def predict_patient(raw_row: dict[str, Any]) -> dict[str, Any]:
    row_with_pc, pc_values = add_pc1(raw_row)
    model_input = make_model_matrix(row_with_pc)
    votes = {_class: 0 for _class in ASSETS["classes"]}
    for tree in ASSETS["rf"]["trees"]:
        votes[_predict_tree(tree, model_input)] += 1
    yes_votes = votes.get(ASSETS["yes_class"], 0)
    probability = yes_votes / float(ASSETS["ntree"])
    threshold = float(ASSETS["threshold"])
    return {
        "probability": probability,
        "predicted_class": "Yes" if probability >= threshold else "No",
        "threshold": threshold,
        "votes": votes,
        "pc_values": pc_values,
        "model_input": model_input,
    }


def build_raw_row(
    *,
    therapy: str,
    gender: str,
    age: float,
    bmi: float,
    fmr: float,
    st_value: str,
    sct_value: str,
    ati: float,
    ka: float,
    act_inputs: dict[str, float],
    bal_inputs: dict[str, float],
) -> dict[str, Any]:
    row = {
        "Type": therapy,
        "Gender": gender,
        "Age": age,
        "BMI": bmi,
        "FMR": fmr,
        "ST": st_value,
        "SCT": sct_value,
        "ATI": ati,
        "KA": ka,
    }
    row.update(act_inputs)
    row.update(bal_inputs)
    return row
