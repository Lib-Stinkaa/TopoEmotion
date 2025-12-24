# Example Visualizations

Academic-style figures demonstrating the TopoEmotion pipeline.

## Figures

**fig1_parameters.png** - Takens embedding parameters (τ and d for each signal)

**fig2_raw_signals.png** - Raw signals at 1000 Hz (8 seconds)

**fig3_preprocessed_signals.png** - Preprocessed signals after filtering

**fig4_downsampled_signals.png** - Downsampled to 100 Hz

**fig5_embeddings.png** - 2D projections of time-delay embeddings

**fig6_fnn_curves.png** - False Nearest Neighbors analysis for dimension selection

**fig7_persistence_landscapes.png** - Persistence landscape features (500 values per signal, H0+H1)

## Generation

```bash
python generate_final_figures.py
```

This script generates all 7 figures (fig1-fig7) using the TopoEmotion pipeline.
