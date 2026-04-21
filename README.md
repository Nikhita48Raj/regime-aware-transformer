Model Comparison

1. PatchTSTSimple Baseline
- Best Val Loss: 0.3012
- MAE: 2.3176
- RMSE: 2.8540
- sMAPE: 32.29%

2. Regime-Aware PatchTST
- Best Val Loss: 0.2664
- MAE: 2.2982
- RMSE: 2.9173
- sMAPE: 31.95%

Observations:
- Regime-aware model improves validation loss, MAE, and sMAPE
- RMSE increases slightly, suggesting some larger forecast errors remain
- Latent regime conditioning appears beneficial but not uniformly across all error types