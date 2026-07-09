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


def _is_numeric_level_marker(levels: Any) -> bool:
    if not isinstance(levels, list):
        return True
    return len(levels) == 1 and str(levels[0]) in {"0", "0.0"}


def _categorical_go_left(value: Any, split_point: float, levels: list[Any]) -> bool:
    levels_str = [str(x) for x in levels]
    value_str = str(value)
    if value_str not in levels_str:
        value_str = levels_str[0]
    level_index = levels_str.index(value_str) + 1
    mask = int(round(float(split_point)))
    return ((mask >> (level_index - 1)) & 1) == 1


def _predict_tree_value(tree: dict[str, list[Any]], x: dict[str, Any], rf_model: dict[str, Any]) -> Any:
    node = 0
    columns = rf_model["predictor_names"]
    xlevels = rf_model.get("xlevels") or {}
    classes = rf_model.get("classes") or []

    while True:
        status = int(tree["status"][node])
        split_var = int(tree["split_var"][node])
        if status == -1 or split_var == 0:
            pred = tree["prediction"][node]
            if rf_model["type"] == "classification":
                pred_idx = int(round(float(pred))) - 1
                return classes[pred_idx]
            return float(pred)

        var = columns[split_var - 1]
        split_point = float(tree["split_point"][node])
        levels = xlevels.get(var)
        if levels is not None and not _is_numeric_level_marker(levels):
            go_left = _categorical_go_left(x.get(var), split_point, levels)
        else:
            go_left = _to_float(x.get(var), 0.0) <= split_point

        node = int(tree["left"][node] if go_left else tree["right"][node]) - 1


def require_complete_raw(raw_row: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce a complete web input row without missing-value imputation."""
    out: dict[str, Any] = {}
    input_types = ASSETS["input_types"]
    factor_levels = ASSETS.get("factor_levels_raw", {})

    for var in ASSETS["defaults"]:
        value = raw_row.get(var)
        if _is_missing(value):
            raise ValueError(f"Missing required input: {var}. The public calculator does not impute missing values.")

        if input_types.get(var) == "numeric":
            try:
                out[var] = float(value)
            except Exception as exc:
                raise ValueError(f"Invalid numeric input for {var}: {value}") from exc
        else:
            value_str = str(value)
            levels = [str(v) for v in factor_levels.get(var, [])]
            if levels and value_str not in levels:
                raise ValueError(f"Invalid categorical input for {var}: {value_str}")
            out[var] = value_str

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


def add_pc1(raw_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    complete_raw = require_complete_raw(raw_row)
    z_row = zscore_row(complete_raw)
    pca_input = complete_raw if ASSETS.get("use_external_pca_helper") else z_row

    act_scores = predict_svd_pca(pca_input, ASSETS["pca_activity_stage1"])
    bal_scores = predict_svd_pca(pca_input, ASSETS["pca_balance_stage1"])
    stage1 = {**act_scores, **bal_scores}
    integrated = predict_svd_pca(stage1, ASSETS["pca_twostage_integrated"])

    row_with_pc = dict(z_row)
    row_with_pc.update(integrated)
    pc_values = {**stage1, **integrated}
    return row_with_pc, pc_values, complete_raw

def _derive_bmi(height: Any, weight: Any) -> float:
    if _is_missing(height) or _is_missing(weight):
        raise ValueError("Height and weight are required to compute BMI. The public calculator does not impute missing values.")
    height_value = float(height)
    weight_value = float(weight)
    if height_value <= 0:
        raise ValueError("Height must be greater than 0 to compute BMI.")
    return weight_value / (height_value ** 2)


def _derive_fmr(fat_mass: Any, muscle_mass: Any) -> float:
    if _is_missing(fat_mass) or _is_missing(muscle_mass):
        raise ValueError("Fat mass and muscle mass are required to compute FMR. The public calculator does not impute missing values.")
    fat_mass_value = float(fat_mass)
    muscle_mass_value = float(muscle_mass)
    if muscle_mass_value <= 0:
        raise ValueError("Muscle mass must be greater than 0 to compute FMR.")
    return fat_mass_value / muscle_mass_value


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
    return str(_predict_tree_value(tree, x, ASSETS["rf"]))


def predict_patient(raw_row: dict[str, Any]) -> dict[str, Any]:
    row_with_pc, pc_values, complete_raw = add_pc1(raw_row)
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
        "complete_raw": complete_raw,
        "model_input": model_input,
    }


def build_raw_row(
    *,
    therapy: str,
    gender: str,
    age: float,
    st_value: str,
    sct_value: str,
    ati: float,
    ka: float,
    act_inputs: dict[str, float],
    bal_inputs: dict[str, float],
    bmi: float | None = None,
    fmr: float | None = None,
    height: float | None = None,
    weight: float | None = None,
    fat_mass: float | None = None,
    muscle_mass: float | None = None,
) -> dict[str, Any]:
    bmi_value = bmi if not _is_missing(bmi) else _derive_bmi(height, weight)
    fmr_value = fmr if not _is_missing(fmr) else _derive_fmr(fat_mass, muscle_mass)
    row = {
        "Type": therapy,
        "Gender": gender,
        "Age": age,
        "BMI": bmi_value,
        "FMR": fmr_value,
        "ST": st_value,
        "SCT": sct_value,
        "ATI": ati,
        "KA": ka,
    }
    row.update(act_inputs)
    row.update(bal_inputs)
    return row
