"""Opt_CWM codebase.

Source:
https://github.com/neuroailab/Opt_CWM/tree/79bd593ac110886352b3b7fbe21746c1dabd30be

The codebase has been minimally modified to work with the rest of the project:
- Added __init__.py (this file)to make the code importable.
- Prefixed all imports with multitasking.models._extern.opt_cwm.
- Removed all images and demo data files.
- Set weights_only=False for checkpoint loading, which is required for more recent
  PyTorch versions.
- Use the scratch directory for caching the model weights.
"""
