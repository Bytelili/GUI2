from __future__ import annotations

import numpy as np

from tn_dpo_gui.scoring.nullspace_projection import orthogonality_dot, project_preference_to_task_nullspace


def test_projection_removes_task_component() -> None:
    rho, null = project_preference_to_task_nullspace([1.0, 2.0], [2.0, 4.0])
    assert np.isclose(rho, 2.0)
    assert np.allclose(null, np.zeros(2))
    assert np.isclose(orthogonality_dot([1.0, 2.0], null), 0.0)
