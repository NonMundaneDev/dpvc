from .wrapper import *
from .anonymizer import *
from .openvoice import OpenVoiceWrapper
from .controlvc import ControlVCWrapper
from . import model_embedding_vae
from .model_embedding_vae import *

__all__ = [
    "ControlVCWrapper",
    "OpenVoiceWrapper",
]

try:
    from .naturalspeech3 import NaturalSpeech3Wrapper

    __all__.append("NaturalSpeech3Wrapper")
except ImportError:
    NaturalSpeech3Wrapper = None

try:
    from .vec2wav2 import Vec2Wav2Wrapper

    __all__.append("Vec2Wav2Wrapper")
except ImportError:
    Vec2Wav2Wrapper = None

__version__ = "0.2.0"
