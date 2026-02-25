"""Unit tests for the MMFlow models."""

import pytest

mmflow = pytest.importorskip("mmflow")


@pytest.mark.parametrize("model_name", mmflow.list_models())
def test_mmflow_model_loading(model_name: str):
    model = mmflow.build_model(model_name)
    assert model is not None
