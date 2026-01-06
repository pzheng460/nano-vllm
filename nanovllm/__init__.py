# Initialize device backend FIRST, before any other imports
# This ensures torch.compile is patched before decorators are evaluated
from nanovllm.utils.device import DeviceBackend
DeviceBackend.initialize()

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
