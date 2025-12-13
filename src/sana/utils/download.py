"""
Functions for downloading pre-trained Sana models
"""
import os
import os.path as osp

import torch
from torchvision.datasets.utils import download_url
from huggingface_hub import hf_hub_download, snapshot_download


pretrained_models = {}


def hf_download_or_fpath(path):
    if osp.exists(path):
        return path

    if path.startswith("hf://"):
        segs = path.replace("hf://", "").split("/")
        repo_id = "/".join(segs[:2])
        filename = "/".join(segs[2:])
        return hf_download_data(repo_id, filename, repo_type="model", download_full_repo=True)


def hf_download_data(
    repo_id="Efficient-Large-Model/Sana_1600M_1024px",
    filename="checkpoints/Sana_1600M_1024px.pth",
    cache_dir=None,
    repo_type="model",
    download_full_repo=False,
):
    """
    Download dummy data from a Hugging Face repository.

    Args:
    repo_id (str): The ID of the Hugging Face repository.
    filename (str): The name of the file to download.
    cache_dir (str, optional): The directory to cache the downloaded file.

    Returns:
    str: The path to the downloaded file.
    """
    try:
        if download_full_repo:
            # download full repos to fit dc-ae
            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                repo_type=repo_type,
            )
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            repo_type=repo_type,
        )
        return file_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None


def find_model(model_name):
    """
    Finds a pre-trained G.pt model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained G.pt checkpoints
        return download_model(model_name)

    # Load a custom Sana checkpoint:
    model_name = hf_download_or_fpath(model_name)
    assert os.path.isfile(model_name), f"Could not find Sana checkpoint at {model_name}"
    return torch.load(model_name, map_location=lambda storage, loc: storage)


def download_model(model_name):
    """
    Downloads a pre-trained Sana model from the web.
    """
    assert model_name in pretrained_models
    local_path = f"output/pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        hf_endpoint = os.environ.get("HF_ENDPOINT")
        if hf_endpoint is None:
            hf_endpoint = "https://huggingface.co"
        os.makedirs("output/pretrained_models", exist_ok=True)
        web_path = f""
        download_url(web_path, "output/pretrained_models/")
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model
