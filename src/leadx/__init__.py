"""leadx — LSH ensemble classifier."""
from importlib.metadata import PackageNotFoundError, version

from .backbone import LeadXBackbone, TaskModel
from .classifier import LeadX
from .sklearn_api import LeadXClassifier

try:
    __version__ = version("leadx")
except PackageNotFoundError:  # editable/source checkout without install metadata
    __version__ = "0.0.0+unknown"
__all__ = ["LeadX", "LeadXClassifier", "LeadXBackbone", "TaskModel"]
