# RF 0.708 Streamlit web calculator

This folder contains the updated Streamlit calculator for the current Random Forest main model.

## Model

- Model: Random Forest main model
- Internal validation AUC mean: 0.708
- Target class: `Yes` / effective improvement
- PCA strategy: two-stage PCA
- Final model PCA input: integrated `PC1` only
- Web input policy: complete input is required for all predictors; no missing-value imputation is deployed in the public calculator

## GitHub files to upload

Upload these files to the root of the GitHub repository:

- `web_app.py`
- `web_spine_app.py`
- `web_app_cn.py`
- `web_app_en.py`
- `rf708_calculator_core.py`
- `model_assets.json`
- `calculator_input_schema.csv`
- `requirements.txt`

## Run locally

Chinese page:

```bash
streamlit run web_app.py
```

English page:

```bash
streamlit run web_spine_app.py
```

Alternative entry names:

```bash
streamlit run web_app_cn.py
streamlit run web_app_en.py
```

## Streamlit Community Cloud

1. Push the files above to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Select the repository.
4. Set the main file path to one of:
   - `web_app.py` for the Chinese calculator
   - `web_spine_app.py` for the English calculator
5. Deploy.
