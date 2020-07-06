import warnings

from torchvision import models
from torchvision import datasets
from torchvision import ops
from torchvision import transforms
from torchvision import utils
from torchvision import io

from .extension import _HAS_OPS
import torch

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass

_image_backend = 'PIL'

_video_backend = "pyav"


def set_image_backend(backend):
    """
    Specifies the package used to load images.

    Args:
        backend (string): Name of the image backend. one of {'PIL', 'accimage'}.
            The :mod:`accimage` package uses the Intel IPP library. It is
            generally faster than PIL, but does not support as many operations.
    """
    global _image_backend
    if backend not in ['PIL', 'accimage']:
        raise ValueError("Invalid backend '{}'. Options are 'PIL' and 'accimage'"
                         .format(backend))
    _image_backend = backend


def get_image_backend():
    """
    Gets the name of the package used to load images
    """
    return _image_backend


def _check_cuda_matches():
    """
    Make sure that CUDA versions match between the pytorch install and torchvision install
    """
    import torch
    from torchvision import _C
    if hasattr(_C, "CUDA_VERSION") and torch.version.cuda is not None:
        tv_version = str(_C.CUDA_VERSION)
        if int(tv_version) < 10000:
            tv_major = int(tv_version[0])
            tv_minor = int(tv_version[2])
        else:
            tv_major = int(tv_version[0:2])
            tv_minor = int(tv_version[3])
        t_version = torch.version.cuda
        t_version = t_version.split('.')
        t_major = int(t_version[0])
        t_minor = int(t_version[1])
        if t_major != tv_major or t_minor != tv_minor:
            raise RuntimeError("Detected that PyTorch and torchvision were compiled with different CUDA versions. "
                               "PyTorch has CUDA Version={}.{} and torchvision has CUDA Version={}.{}. "
                               "Please reinstall the torchvision that matches your PyTorch install."
                               .format(t_major, t_minor, tv_major, tv_minor))


_check_cuda_matches()
def set_video_backend(backend):
    """
    Specifies the package used to decode videos.

    Args:
        backend (string): Name of the video backend. one of {'pyav', 'video_reader'}.
            The :mod:`pyav` package uses the 3rd party PyAv library. It is a Pythonic
            binding for the FFmpeg libraries.
            The :mod:`video_reader` package includes a native C++ implementation on
            top of FFMPEG libraries, and a python API of TorchScript custom operator.
            It is generally decoding faster than :mod:`pyav`, but perhaps is less robust.
    """
    global _video_backend
    if backend not in ["pyav", "video_reader"]:
        raise ValueError(
            "Invalid video backend '%s'. Options are 'pyav' and 'video_reader'" % backend
        )
    if backend == "video_reader" and not io._HAS_VIDEO_OPT:
        warnings.warn("video_reader video backend is not available")
    else:
        _video_backend = backend


def get_video_backend():
    return _video_backend


def _is_tracing():
    return torch._C._get_tracing_state()
