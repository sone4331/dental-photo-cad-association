# Dental photograph-to-CAD association

This archive supports **Associating dental component photographs with computer-aided design models using known batch inventories**. The main pipeline uses direct appearance descriptors, silhouette scores, inventory-wise normalization, equal-weight fusion and one-to-one assignment. Residual mapping is retained only as a comparison.

## Source map

| Directory or file | Function |
|---|---|
| `source/modules/appearance/run_baseline.py` | DINOv2-SALAD model, CAD-source training and appearance scoring |
| `source/modules/adaptation/controlled_train.py` | Photograph adaptation, CAD replay and appearance randomization |
| `source/modules/pipeline/` | Training orchestration, control analyses, verification and timing |
| `source/modules/evaluation/` | Direct-descriptor comparisons, paired intervals and online timing |
| `source/modules/assignment/assignment.py` | Independent, greedy and global decision rules |
| `source/modules/silhouette/geometry_baseline.py` | Foreground normalization, silhouette descriptors and distance scoring |
| `source/modules/render_gallery.py` | CAD rendering; the cost driver supplies the 42-view camera configuration |
| `method_configuration.json` | Initialization, training, scoring and evaluation settings |
| `source_manifest.json` | SHA-256 checksums of the supplied Python files |

The pipeline directory includes the assignment implementation used by its analysis driver. Directory names describe functions. Numerical operations are preserved; path strings have been adjusted to this layout. A manuscript-file protection check unrelated to the computations has been omitted.

## Environment and external dependencies

Recorded environment: Python 3.12.3; PyTorch 2.14.0+cu126; NumPy 2.5.2; SciPy 1.18.1; OpenCV-headless 5.0.0.93; pytorch-metric-learning 2.9.0; trimesh 5.1.0; pyrender 0.1.45; PyOpenGL 3.1.0; and Pillow. These are recorded versions, not a newly tested installation recipe.

### DINOv2

Use the public non-register ViT-L/14 backbone from the [official DINOv2 repository](https://github.com/facebookresearch/dinov2).

| Environment variable | Required value |
|---|---|
| `DINOV2_REPO` | Local DINOv2 checkout directory |
| `DINOV2_PRETRAINED` | Path to the public backbone weights |
    
Backbone weight filename:

```text
dinov2_vitl14_pretrain.pth
```

### SALAD

Obtain the aggregation implementation from the [official SALAD repository](https://github.com/serizba/salad).

| Item | Value |
|---|---|
| Commit | `6aede13a3f6c25750bf7fde10209c06cb73060bb` |
| File in the official repository | `models/aggregators/salad.py` |
| Required location in this code package | `source/modules/appearance/official_source/models/aggregators/salad.py` |

Recorded SHA-256 of the SALAD module:

```text
7c24560a1653a167aeea248f4566b445aa5a39dacc252f13adbac2f1a4f68726
```

SALAD uses GPL-3.0. Obtain third-party dependencies and weights from their official repositories and retain their applicable licenses. They are not redistributed here, and this archive does not grant an alternative license to them.

## Inputs and execution conditions

This is a **code-only archive**. It does not contain company photographs, STL models, row-level pairing records, rendered banks, geometry/lighting caches, prediction caches or trained checkpoints. Reproducing the reported numbers requires those original resources; this archive alone is not a self-contained reproduction package.

Code and supporting resources are resolved relative to the source root. The paths below describe the expected layout; resource directories are not included in this code-only archive.

| Path | Purpose |
|---|---|
| `source/` | Source root |
| `source/modules/` | Implementations and supporting resource locations |
| `source/modules/score_cache/` | Cached scores |
| `source/modules/inventories/` | Inventory definitions and cases |
| `source/modules/data_groups/` | Grouping metadata |
| `source/modules/geometry_cache/` | Geometry resources |
| `source/modules/cache/` | Cached reference resources |
| `source/dataset/models/` | Authorized raw CAD files |

Resource manifests and their path keys must match this layout. Source-check manifests must use the packaged checksums. Internal provenance mapping is retained by the authors.

Protocol fields are grouped by their role:

| Structure | Field names |
|---|---|
| Photograph records | `query_id`<br>`model_id`<br>`group_id`<br>`outer_fold`<br>`repaired_path` |
| Gallery and training settings | `gallery_model_ids`<br>`training_seeds`<br>`primary_seed`<br>`source_train_views`<br>`execution_order` |
| Inventory definitions | `batch_id`<br>`fold`<br>`layout`<br>`cohort`<br>`capacity`<br>`seed`<br>`candidate_cad_ids`<br>`full_capacity_batch` |
| Cases | `case_id`<br>`batch_id`<br>`scenario`<br>`query_ids` |

The photograph-path field must resolve to the authorized photograph resource. Inventory definitions also record count metadata. Reference identities are used for evaluation; assignment receives only candidate scores and does not receive the true photograph-to-design mapping.

Read the [method configuration](method_configuration.json) for the four-epoch CAD-source stage, 20-epoch photograph adaptation, fold isolation, scoring rules and resampling settings. Obtain the original authorized resources and matching protocol files before running the experiment drivers. Do not interpret a missing-resource error as a new experiment result.

Experimental data access requires the provider's prior approval and may be requested through the corresponding author. Approval for third-party disclosure during review has not yet been obtained. Requests for derived resources or checkpoints can also be directed to the corresponding author; their availability is not implied by this code link.

## Download

Use **Code > Download ZIP** on this repository, or clone it:

```sh
git clone https://github.com/sone4331/dental-photo-cad-association.git
```

The repository contains source code and method settings. Data and trained model resources are not included.

