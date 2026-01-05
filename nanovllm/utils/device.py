import torch


class DeviceType:
    CUDA = "cuda"
    NPU = "npu"


class DeviceBackend:
    _instance = None
    _device_type = None

    @classmethod
    def detect_device(cls):
        try:
            import torch_npu
            if torch.npu.is_available():
                return DeviceType.NPU
        except ImportError:
            pass
        if torch.cuda.is_available():
            return DeviceType.CUDA
        raise RuntimeError("No GPU/NPU device available")

    @classmethod
    def initialize(cls, device_type=None):
        if cls._instance is not None:
            return cls._instance
        cls._device_type = device_type or cls.detect_device()
        if cls._device_type == DeviceType.NPU:
            import torch_npu  # noqa
        cls._instance = cls()
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("DeviceBackend not initialized. Call DeviceBackend.initialize() first.")
        return cls._instance

    @property
    def is_npu(self):
        return self._device_type == DeviceType.NPU

    @property
    def is_cuda(self):
        return self._device_type == DeviceType.CUDA

    @property
    def device_name(self):
        return self._device_type

    def set_device(self, rank):
        if self.is_cuda:
            torch.cuda.set_device(rank)
        else:
            torch.npu.set_device(rank)

    def synchronize(self):
        if self.is_cuda:
            torch.cuda.synchronize()
        else:
            torch.npu.synchronize()

    def empty_cache(self):
        if self.is_cuda:
            torch.cuda.empty_cache()
        else:
            torch.npu.empty_cache()

    def reset_peak_memory_stats(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        else:
            torch.npu.reset_peak_memory_stats()

    def mem_get_info(self):
        if self.is_cuda:
            return torch.cuda.mem_get_info()
        else:
            return torch.npu.mem_get_info()

    def memory_stats(self):
        if self.is_cuda:
            return torch.cuda.memory_stats()
        else:
            return torch.npu.memory_stats()

    def get_dist_backend(self):
        return "nccl" if self.is_cuda else "hccl"

    def to_device(self, tensor, non_blocking=True):
        if self.is_cuda:
            return tensor.cuda(non_blocking=non_blocking)
        else:
            return tensor.npu(non_blocking=non_blocking)


def get_device_backend():
    return DeviceBackend.get_instance()


def is_npu():
    return DeviceBackend._instance is not None and DeviceBackend._instance.is_npu


def is_cuda():
    return DeviceBackend._instance is not None and DeviceBackend._instance.is_cuda
