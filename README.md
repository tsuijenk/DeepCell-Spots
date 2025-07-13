# DeepCell-Spots (Custom Fork for Cluster Use)

This is a customized fork of `deepcell-spots` for running DeepCell spot detection in parallel on compute clusters (e.g., SLURM arrays).

---

## Custom Patch: Modified `spot_detection.py`

This fork includes a modification to `deepcell_spots/applications/spot_detection.py` to prevent issues when extracting the model archive concurrently in job arrays.

### What Changed?

- The model is no longer auto-extracted by every job.
- Prevents file corruption like `PermissionError` and `Read less bytes than requested`.
- Compatible with SLURM parallel processing of FOV/Z combinations.

---

## Bypass DeepCell Access Token

Instead of requiring a `DEEPCELL_ACCESS_TOKEN`, **pre-extract the model manually**:

```bash
mkdir -p ~/.deepcell/models/SpotDetection-8
tar -xzf ~/.deepcell/models/SpotDetection-8.tar.gz -C ~/.deepcell/models/SpotDetection-8
```

---

## Running `deepcell_spots_experiment`

### 1. Clone This Fork

```bash
git clone https://github.com/tsuijenk/DeepCell-Spots.git
cd DeepCell-Spots
```

Ensure your environment uses this version of the package.

### 2. SLURM Execution

#### Run DeepCell Spots Per-FOV/Z (Non-Preprocessed Images):

```bash
sbatch ./experimental_run_script/polaris_run_both_xps.sh
```

#### Merge Output per Dataset:

```bash
sbatch ./experimental_run_script/batch_merge_polaris_output.sh
```

---

## Note on Model Tarball and Permissions

You **do not need to change file permissions** on `SpotDetection-8.tar.gz`.

In this patched version of `deepcell-spots`, the model is **not auto-extracted** if the extracted folder already exists. This avoids concurrent extraction issues.

To set it up correctly:

```bash
mkdir -p ~/.deepcell/models/SpotDetection-8
tar -xzf ~/.deepcell/models/SpotDetection-8.tar.gz -C ~/.deepcell/models/SpotDetection-8
```

Make sure the following files and directories exist inside `~/.deepcell/models/SpotDetection-8/`:

- `saved_model.pb`
- `keras_metadata.pb`
- `variables/`
- `assets/`

No access token is required once the model is extracted.
