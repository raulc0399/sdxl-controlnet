# adapted from a1111

import os
from typing import Callable
import torch
import spandrel
from PIL import Image
import numpy as np
import tqdm

import upscaler.image_grid as image_grid

class UpscalerRealESRGAN:
    def __init__(self, models_path, model_name, prefer_half=False):
        self.name = "RealESRGAN"
        self.models_path = models_path
        self.model_name = model_name
        self.model_descriptor = None
        self.device = "cuda"
        self.prefer_half = prefer_half

    def load_model(self):
        local_data_path = os.path.join(self.models_path, f"{self.model_name}.pth")

        self.model_descriptor = self.load_spandrel_model(
            local_data_path,
            device=self.device,
            prefer_half=self.prefer_half,
            expected_architecture="ESRGAN",  # "RealESRGAN" isn't a specific thing for Spandrel
        )
    
    def load_spandrel_model(
        self,
        path: str | os.PathLike,
        *,
        device: str | torch.device | None,
        prefer_half: bool = False,
        dtype: str | torch.dtype | None = None,
        expected_architecture: str | None = None,
    ) -> spandrel.ModelDescriptor:
        model_descriptor = spandrel.ModelLoader(device=device).load_from_file(str(path))
        if expected_architecture and model_descriptor.architecture != expected_architecture:
            print(
                f"Model {path!r} is not a {expected_architecture!r} model (got {model_descriptor.architecture!r})",
            )
        half = False
        if prefer_half:
            if model_descriptor.supports_half:
                model_descriptor.model.half()
                half = True
            else:
                print(f"Model {path} does not support half precision, ignoring --half")
        if dtype:
            model_descriptor.model.to(dtype=dtype)
        model_descriptor.model.eval()
        print(
            f"Loaded {model_descriptor} from {path} (device={device}, half={half}, dtype={dtype})"
        )
        return model_descriptor

    def do_upscale(self, img, tile_size=0, tile_overlap=0):
        if self.model_descriptor is None:
            self.load_model()

        return self.upscale_with_model(
            self.model_descriptor,
            img,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    
    def pil_image_to_torch_bgr(self, img: Image.Image) -> torch.Tensor:
        img = np.array(img.convert("RGB"))
        img = img[:, :, ::-1]  # flip RGB to BGR
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
        return torch.from_numpy(img)

    def torch_bgr_to_pil_image(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.ndim == 4:
            # If we're given a tensor with a batch dimension, squeeze it out
            # (but only if it's a batch of size 1).
            if tensor.shape[0] != 1:
                raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
            tensor = tensor.squeeze(0)
        assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
        # TODO: is `tensor.float().cpu()...numpy()` the most efficient idiom?
        arr = tensor.float().cpu().clamp_(0, 1).numpy()  # clamp
        arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
        arr = arr.round().astype(np.uint8)
        arr = arr[:, :, ::-1]  # flip BGR to RGB
        return Image.fromarray(arr, "RGB")

    def upscale_pil_patch(self, model, img: Image.Image) -> Image.Image:
        """
        Upscale a given PIL image using the given model.
        """
        param = self.get_param(model)

        with torch.no_grad():
            tensor = self.pil_image_to_torch_bgr(img).unsqueeze(0)  # add batch dimension
            tensor = tensor.to(device=param.device, dtype=param.dtype)
            # with torch.autocast("cuda"):
            return self.torch_bgr_to_pil_image(model(tensor))
    
    def upscale_with_model(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        img: Image.Image,
        *,
        tile_size: int,
        tile_overlap: int = 0,
        desc="tiled upscale",
    ) -> Image.Image:
        if tile_size <= 0:
            print("Upscaling without tiling", img)
            output = self.upscale_pil_patch(model, img)
            print("=> ", output)
            return output

        grid = image_grid.split_grid(img, tile_size, tile_size, tile_overlap)
        newtiles = []

        with tqdm.tqdm(total=grid.tile_count, desc=desc, disable=False) as p:
            for y, h, row in grid.tiles:
                newrow = []
                for x, w, tile in row:
                    # if shared.state.interrupted:
                    #     return img
                    output = self.upscale_pil_patch(model, tile)
                    scale_factor = output.width // tile.width
                    newrow.append([x * scale_factor, w * scale_factor, output])
                    p.update(1)
                newtiles.append([y * scale_factor, h * scale_factor, newrow])

        newgrid = image_grid.Grid(
            newtiles,
            tile_w=grid.tile_w * scale_factor,
            tile_h=grid.tile_h * scale_factor,
            image_w=grid.image_w * scale_factor,
            image_h=grid.image_h * scale_factor,
            overlap=grid.overlap * scale_factor,
        )
        return image_grid.combine_grid(newgrid)
    
    def get_param(self, model) -> torch.nn.Parameter:
        """
        Find the first parameter in a model or module.
        """
        if hasattr(model, "model") and hasattr(model.model, "parameters"):
            # Unpeel a model descriptor to get at the actual Torch module.
            model = model.model

        for param in model.parameters():
            return param

        raise ValueError(f"No parameters found in model {model!r}")
