# HydroSat Final Technical Proposal

## Final Concept

HydroSat is an onboard water-quality intelligence pipeline, not only a regression model. The final proposed system combines a stable CPU-first global ensemble with an optical-regime decision layer that adds water-type awareness, uncertainty estimation, and selective downlink logic.

## Final Architecture

```text
Multispectral image patch
 -> quality gate
 -> spectral, index, and texture feature extraction
 -> stable global ensemble prediction
 -> optical-regime router
 -> regime expert correction
 -> uncertainty and disagreement estimation
 -> calibrated final output
 -> selective downlink decision
```

## Prediction Model

The final technical solution is a hybrid model:

- A deterministic global ensemble provides the primary turbidity and chlorophyll-a estimates.
- An optical-regime router classifies observations into water-type regimes.
- Regime-specific expert models provide calibrated corrections.
- Model disagreement is used as an uncertainty signal.
- Low-confidence or out-of-distribution observations are flagged for selective image-patch downlink.

The best local hybrid validation used:

```text
turbidity_final = 0.80 * global_turbidity + 0.20 * regime_turbidity
chla_final      = 0.97 * global_chla      + 0.03 * regime_chla
```

## Local Validation Evidence

The best previous global ensemble validation score was:

```text
Algorithm score: 57.1568
```

The hybrid regime-aware sweep improved this to:

```text
Turbidity score: 54.5458
Chl-a score:     60.7303
Algorithm score: 57.6381
Improvement:     +0.4813
```

This shows that the optical-regime layer is not only useful for explainability and confidence. A small regime-aware correction also improved local validation while preserving the stability of the global ensemble.

## Official Local Scoring Check

The released local Area8 truth files contain:

```text
Turbidity truth points: 372
Chl-a truth points:     103
Required Area8 scenes:  59
```

The available Kaggle image bundle contained only 1 of the 59 required Area8 TIFF scenes. Therefore, complete image-based local scoring could not be reproduced without the missing 58 scenes. The official scoring formula and key matching were still implemented and verified.

When the inference package fell back to constant public statistics because the images were missing, the full-truth fallback score was:

```text
Turbidity score: 0.0
Chl-a score:     0.0
Algorithm score: 0.0
```

This confirms that fallback-only inference is not a viable final model and that the meaningful model evidence comes from the hybrid validation sweep.

## Innovation and Mission Plausibility

The proposal is strongest when framed as a spacecraft decision system:

- It performs quality gating before regression.
- It uses target-specific spectral features for turbidity and chlorophyll-a.
- It separates stable prediction from optical-regime interpretation.
- It quantifies uncertainty through model disagreement.
- It supports selective downlink of compact prediction packets, masks, thumbnails, or high-value image patches.
- It remains CPU-executable, with optional neural extensions for later missions.

## Final Claim

HydroSat moves water-quality analysis closer to the sensor. The final proposed system performs regime-aware spectral inference onboard, produces calibrated turbidity and chlorophyll-a estimates, attaches confidence and quality flags, and prioritizes only the most decision-relevant products for downlink.

