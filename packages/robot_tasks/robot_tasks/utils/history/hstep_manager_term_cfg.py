from __future__ import annotations

from isaaclab.managers.manager_term_cfg import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass


@configclass
class HStepObservationTermCfg(ObservationTermCfg):
    history_step: int = 1


@configclass
class HStepObservationGroupCfg(ObservationGroupCfg):
    history_step: int = 1
