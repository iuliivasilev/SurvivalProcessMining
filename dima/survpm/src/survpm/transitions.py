"""Прежний путь импорта; реализация — :mod:`survpm.models.transitions`."""
from .models.transitions import (TransitionModel, TransitionPicker, MinTimePicker,  # noqa: F401
                                ExponentialTransitionModel, register_transition_model,
                                create_transition_model, _TRANSITION_MODELS)
