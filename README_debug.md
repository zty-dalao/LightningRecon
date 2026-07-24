# Debug notes

## Root cause
The training loop previously assumed every sample had exactly 491 views and used a hard-coded `V_total = 491` when sampling and indexing projections. In practice the dataset contains cases with 316-492 available projections, so `torch.index_select` could receive out-of-bounds indices and trigger CUDA device-side asserts.

## Fix applied
- Added `subsample_projections()` in [src/train.py](src/train.py) to sample using the actual number of views present in the current tensor.
- The training loop now derives `actual_views = projs.shape[1]` and uses that for subsampling instead of the hard-coded 491.
- Added a regression test in [tests/test_dataset.py](tests/test_dataset.py) to cover this case.

## Verification
- `python -m unittest -q tests.test_dataset` ✅
- A real forward pass on a dataset sample with `subsample_projections(..., 24, ...)` succeeded and produced an output tensor. ✅
