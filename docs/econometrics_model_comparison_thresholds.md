# Model Comparison Thresholds

The canonical machine-readable configuration is `config/model_comparison_thresholds.yaml`. The artifact builder copies the exact file into the versioned frontend bundle and records its SHA-256 hash.

The defaults are generic project thresholds, not tuned to feature direction, statistical significance, or client desirability.

Key defaults:

- minimum stable baseline magnitude: 1.0 percentage point;
- stable magnitude tolerance: 1.0 percentage point;
- substantial attenuation/amplification: at least 2.0 percentage points and 40% in relative magnitude;
- meaningful sign flip: both estimates at least 1.0 percentage point in magnitude;
- large sample change: 10%;
- large CI-width change: 25%;
- low support: fewer than 100 rows, 30 prompts, 50 URLs, or 20 domains;
- covariance inference warning: standard-error ratio of at least 1.25.

Relative change is unavailable when the baseline is below its minimum or interpretation units differ. Changing these thresholds requires a reviewed configuration revision and artifact rebuild; Streamlit cannot modify them.
