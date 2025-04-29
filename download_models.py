from diffsynth import download_models
from modelscope import snapshot_download

download_models(["FLUX.1-dev"])
snapshot_download('DiffSynth-Studio/Nexus-Gen', local_dir='models/Nexus-Gen')
