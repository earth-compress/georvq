"""georvq.compat: loading legacy research checkpoints."""
import pytest
import torch


def test_remap_rejects_unknown_keys():
    from georvq.compat import remap_state_dict
    with pytest.raises(KeyError):
        remap_state_dict({"mystery.weight": torch.zeros(1)})
