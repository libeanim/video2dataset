"""
resolution subsampler adjusts the resolution of the videos to some constant value
"""
import os
import ffmpeg
import tempfile
from typing import Literal
from decord import VideoReader, cpu, bridge
from deepsparse import Pipeline
from huggingface_hub import snapshot_download

from .subsampler import Subsampler

bridge.set_bridge("torch")
class ClipEmbeddingSubsampler(Subsampler):
    """

    """

    def __init__(
        self,
        batch_size: int = 6,
    ):

        # Download the model from HF
        model_folder = snapshot_download(repo_id="neuralmagic/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K-quant-ds")

        self.image_embed_pipeline = Pipeline.create(
            task="clip_visual",
            model_path=os.path.join(model_folder, "visual.onnx"),
            batch_size=batch_size,
        ) 


    def __call__(self, streams, metadata=None):
        video_bytes = streams["video"]
        subsampled_bytes = []
        clip_embeddings = []
        for vid_bytes in video_bytes:
            reader = VideoReader(vid_bytes, ctx=cpu(0))
            ce = self.image_embed_pipeline(reader[:]).image_embeddings  # type: ignore[attr-defined]
            clip_embeddings.append(ce.numpy().tobytes())

        streams["clip.npy"] = subsampled_bytes
        return streams, metadata, None
