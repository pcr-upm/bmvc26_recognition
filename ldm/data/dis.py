import os
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as Ftrans
from PIL import Image

# =============================
# SHAPES3D DATASET
# ==========================

# Parent directory containing the shapes3d/ folder.
# The final H5 path is built as: SHAPES3D_ROOT/shapes3d/3dshapes.h5
SHAPES3D_ROOT = "<PATH_TO_DISENTANGLEMENT_LIB_ROOT>"
# SHAPES3D_ROOT = "/home/jelias/data/disentanglement_lib/"
SHAPES3D_SPLITS_DIR = "<SHAPES3D_SPLITS_DIR>"
# SHAPES3D_SPLITS_DIR = "/home/jelias/data/disentanglement_lib/shapes3d/splits/"

SHAPES3D_TRAIN_UNBIASED_SPLIT = os.path.join(
    SHAPES3D_SPLITS_DIR,
    "train_unbiased.npy",
)

SHAPES3D_TRAIN_MULTI_BIASED_SPLIT = os.path.join(
    SHAPES3D_SPLITS_DIR,
    "train_multi_biased.npy",
)

SHAPES3D_TEST_FIXED_SPLIT = os.path.join(
    SHAPES3D_SPLITS_DIR,
    "test_fixed.npy",
)

# Updated version to load te required dataset for the experiments. 
class Shapes3D(Dataset):
    def __init__(
        self,
        path,
        original_resolution=64,
        split=None,
        indices_file=None,
        indices=None,
        as_tensor: bool = True,
        do_normalize: bool = True,
        return_labels: bool = False,
        return_index: bool = False,
        **kwargs
    ):
        self.original_resolution = original_resolution
        self.return_labels = return_labels
        self.return_index = return_index

        h5_path = os.path.join(path, "shapes3d", "3dshapes.h5")
        with h5py.File(h5_path, "r") as f:
            self.data = f["images"][:]
            self.labels = f["labels"][:]   # (480000, 6)

        # ------------------------------------------------------------------
        # Split handling
        # ------------------------------------------------------------------
        if indices is not None and indices_file is not None:
            raise ValueError("Use either 'indices' or 'indices_file', not both.")

        if indices_file is not None:
            self.indices = np.load(indices_file)
        elif indices is not None:
            self.indices = np.asarray(indices)
        elif split is None:
            self.indices = np.arange(len(self.data))
        else:
            raise NotImplementedError(
                "Use indices_file or indices for Shapes3D splits."
            )

        self.length = len(self.indices)

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        real_index = int(self.indices[i])

        img = self.data[real_index]
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)

        out = {
            "image": img,
            "labels": self.labels[real_index].astype(np.float32),
            "index": real_index,
        }

        return out
    

# Training with all the dataset
class Shapes3DEntireDataset(Shapes3D):
    def __init__(self, **kwargs):
        super().__init__(path=SHAPES3D_ROOT,
                original_resolution=None,
                **kwargs)

# We create different wrappers for the different versions of the dataset
class Shapes3DTrainUnbiased(Shapes3D):
    def __init__(self, **kwargs):
        super().__init__(
            path=SHAPES3D_ROOT,
            original_resolution=None,
            indices_file=SHAPES3D_TRAIN_UNBIASED_SPLIT,
            **kwargs
        )


class Shapes3DTrainMultiBiased(Shapes3D):
    def __init__(self, **kwargs):
        super().__init__(
            path=SHAPES3D_ROOT,
            original_resolution=None,
            indices_file=SHAPES3D_TRAIN_MULTI_BIASED_SPLIT,
            **kwargs
        )

class Shapes3DTestFixed(Shapes3D):
    def __init__(self, **kwargs):
        super().__init__(
            path=SHAPES3D_ROOT,
            original_resolution=None,
            indices_file=SHAPES3D_TEST_FIXED_SPLIT,
            **kwargs
        )

# ==========================================================================================
# CELEBA DATASET
# ==========================================================================================


# Directory containing list_attr_celeba.txt, list_bbox_celeba.txt and images/
CELEBA_ROOT = "<PATH_TO_CELEBA_ROOT>"
# CELEBA_ROOT = "/home/database/recognition/faces/celeba/"
CELEBA_SPLIT_DIR = "<CELEBA_SPLITS_DIR>"
# CELEBA_SPLIT_DIR = "/home/jelias/data/disentanglement_lib/celeba/splits"

CELEBA_TRAIN_UNBIASED_SPLIT = os.path.join(
    CELEBA_SPLIT_DIR,
    "train_unbiased.npy",
)

CELEBA_TRAIN_EYEGLASSES_MALE_HAT_BIASED_SPLIT = os.path.join(
    CELEBA_SPLIT_DIR,
    "train_eyeglasses_male_hat_biased.npy",
)

CELEBA_TEST_FIXED_SPLIT = os.path.join(
    CELEBA_SPLIT_DIR,
    "test_fixed.npy",
)


class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return Ftrans.crop(
            img,
            self.x1,
            self.y1,
            self.x2 - self.x1,
            self.y2 - self.y1,
        )

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)


class CelebA(Dataset):
    def __init__(
        self,
        path,
        image_size=64,
        indices_file=None,
        indices=None,
        as_tensor=True,
        do_normalize=True,
        return_index=True,
        verify_images=True,
        crop_mode="bbox",          # "bbox", "center", "d2c"
        bbox_margin=0.30,          # 30% margin
        image_dir_name="images",   # "images" or "images_cropped_6464"
        already_cropped=False,     # True if images are already cropped to 64x64
    ):
        self.path = path
        self.image_dir = os.path.join(path, image_dir_name)
        attr_path = os.path.join(path, "list_attr_celeba.txt")
        bbox_path = os.path.join(path, "list_bbox_celeba.txt")

        self.image_size = image_size
        self.return_index = return_index
        self.crop_mode = crop_mode
        self.bbox_margin = bbox_margin
        self.already_cropped = already_cropped
        self.image_dir_name = image_dir_name

        selected_attrs = [
            "Eyeglasses",
            "Male",
            "Wearing_Hat",
            "Smiling",
            "Mustache",
            "Wearing_Earrings",
        ]

        # ------------------------------------------------------------------
        # Reed labels from list_attr_celeba.txt
        # ------------------------------------------------------------------
        with open(attr_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        declared_num = int(lines[0])
        all_attr_names = lines[1].split()
        attr_to_idx = {a: i for i, a in enumerate(all_attr_names)}

        missing_attrs = [a for a in selected_attrs if a not in attr_to_idx]
        if missing_attrs:
            raise ValueError(f"Faltan atributos en list_attr_celeba.txt: {missing_attrs}")

        selected_cols = [attr_to_idx[a] for a in selected_attrs]

        # ------------------------------------------------------------------
        # Read bboxes only if we are going to do online crop
        # ------------------------------------------------------------------
        self.bboxes = {}

        if crop_mode == "bbox" and not already_cropped:
            with open(bbox_path, "r") as f:
                bbox_lines = [line.strip() for line in f if line.strip()]

            declared_bbox_num = int(bbox_lines[0])

            if declared_bbox_num != declared_num:
                raise ValueError(
                    f"El número de muestras en bbox ({declared_bbox_num}) "
                    f"no coincide con attr ({declared_num})."
                )

            for line in bbox_lines[2:]:
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Línea bbox mal formada: {line}")

                filename = parts[0]
                x1 = int(parts[1])
                y1 = int(parts[2])
                w = int(parts[3])
                h = int(parts[4])

                self.bboxes[filename] = (x1, y1, w, h)

            invalid = []
            for fname, (x1, y1, w, h) in self.bboxes.items():
                if w <= 0 or h <= 0:
                    invalid.append((fname, x1, y1, w, h))

            print(f"BBoxes inválidos: {len(invalid)}")
            print("Primeros ejemplos:", invalid[:10])

        # ------------------------------------------------------------------
        # Build aligned samples by filename according to list_attr_celeba.txt
        # ------------------------------------------------------------------
        samples = []

        for line in lines[2:]:
            parts = line.split()
            filename = parts[0]
            raw_attrs = [int(x) for x in parts[1:]]

            labels = [raw_attrs[c] for c in selected_cols]
            labels = ((np.array(labels, dtype=np.int64) + 1) // 2).astype(np.float32)

            if verify_images:
                img_path = os.path.join(self.image_dir, filename)
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"No existe la imagen anotada: {img_path}")

            if crop_mode == "bbox" and not already_cropped and filename not in self.bboxes:
                raise KeyError(f"No existe bbox para la imagen: {filename}")

            samples.append((filename, labels))

        if len(samples) != declared_num:
            raise ValueError(
                f"Cabecera: {declared_num} imágenes, leídas: {len(samples)}"
            )

        self.samples = samples

        # ------------------------------------------------------------------
        # Split handling
        # ------------------------------------------------------------------
        if indices is not None and indices_file is not None:
            raise ValueError("Usa indices o indices_file, no ambos.")

        if indices_file is not None:
            self.indices = np.load(indices_file).astype(np.int64)
        elif indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)
        else:
            self.indices = np.arange(len(self.samples), dtype=np.int64)

        if np.any(self.indices < 0) or np.any(self.indices >= len(self.samples)):
            raise IndexError("Hay índices fuera de rango.")

        # ------------------------------------------------------------------
        # Final transform
        # ------------------------------------------------------------------
        transform = []

        # If we use images_cropped_6464, the images are already cropped and resized
        # So, we don´t resize again
        if not already_cropped:
            transform.append(transforms.Resize((image_size, image_size)))

        if as_tensor:
            transform.append(transforms.ToTensor())

        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )

        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.indices)

    def _crop_with_bbox(self, img, filename):
        x1, y1, w, h = self.bboxes[filename]

        if w <= 0 or h <= 0:
            raise ValueError(
                f"BBox inválido para {filename}: x1={x1}, y1={y1}, w={w}, h={h}"
            )

        margin_x = int(round(w * self.bbox_margin))
        margin_y = int(round(h * self.bbox_margin))

        crop_x1 = x1 - margin_x
        crop_y1 = y1 - margin_y
        crop_x2 = x1 + w + margin_x
        crop_y2 = y1 + h + margin_y

        img_w, img_h = img.size

        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(img_w, crop_x2)
        crop_y2 = min(img_h, crop_y2)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise ValueError(
                f"BBox inválido tras aplicar margen para {filename}: "
                f"raw_bbox={(x1, y1, w, h)}, img_size={(img_w, img_h)}, "
                f"crop=({crop_x1}, {crop_y1}, {crop_x2}, {crop_y2})"
            )

        return img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def _center_crop_fallback(self, img):
        resize_size = int(round(self.image_size * 1.2))
        img = transforms.Resize(resize_size)(img)
        img = transforms.CenterCrop(self.image_size)(img)
        return img

    def _apply_crop(self, img, filename):
        if self.crop_mode == "bbox":
            try:
                return self._crop_with_bbox(img, filename)
            except Exception:
                return self._center_crop_fallback(img)

        elif self.crop_mode == "center":
            return self._center_crop_fallback(img)

        elif self.crop_mode == "d2c":
            return d2c_crop()(img)

        else:
            raise ValueError(f"crop_mode desconocido: {self.crop_mode}")

    def __getitem__(self, i):
        real_index = int(self.indices[i])
        filename, labels = self.samples[real_index]

        img_path = os.path.join(self.image_dir, filename)
        img = Image.open(img_path).convert("RGB")

        if not self.already_cropped:
            img = self._apply_crop(img, filename)

        img = self.transform(img).permute(1, 2, 0)  # HWC for DisDiff

        out = {
            "image": img,
            "labels": labels,
        }

        if self.return_index:
            out["index"] = real_index
            out["filename"] = filename

        return out


class CelebAEntireDataset(CelebA):
    def __init__(self, **kwargs):
        super().__init__(
            path=CELEBA_ROOT,
            image_size=64,
            crop_mode="bbox",
            bbox_margin=0.30,
            **kwargs,
        )


# ==========================================================================================
# CELEBA SPLIT WRAPPERS FOR BIAS EXPERIMENTS
# Original wrappers: use original images and apply bbox crop online.
# ==========================================================================================

def _set_celeba_default_kwargs(kwargs):
    kwargs.setdefault("image_size", 64)
    kwargs.setdefault("crop_mode", "bbox")
    kwargs.setdefault("bbox_margin", 0.30)
    kwargs.setdefault("return_index", True)
    return kwargs


class CelebATrainUnbiased(CelebA):
    def __init__(self, **kwargs):
        kwargs = _set_celeba_default_kwargs(kwargs)
        super().__init__(
            path=CELEBA_ROOT,
            indices_file=CELEBA_TRAIN_UNBIASED_SPLIT,
            **kwargs,
        )


class CelebATrainEyeglassesMaleHatBiased(CelebA):
    def __init__(self, **kwargs):
        kwargs = _set_celeba_default_kwargs(kwargs)
        super().__init__(
            path=CELEBA_ROOT,
            indices_file=CELEBA_TRAIN_EYEGLASSES_MALE_HAT_BIASED_SPLIT,
            **kwargs,
        )


class CelebATestFixed(CelebA):
    def __init__(self, **kwargs):
        kwargs = _set_celeba_default_kwargs(kwargs)
        super().__init__(
            path=CELEBA_ROOT,
            indices_file=CELEBA_TEST_FIXED_SPLIT,
            **kwargs,
        )


# ==========================================================================================
# CELEBA CROPPED 64x64 SPLIT WRAPPERS FOR BIAS EXPERIMENTS. Imagges already cropped for faster executions. 
# New wrappers: use precomputed crops in images_cropped_6464 and skip online crop/resize.
# ==========================================================================================

def _set_celeba_cropped_default_kwargs(kwargs):
    kwargs.setdefault("image_size", 64)
    kwargs.setdefault("crop_mode", "bbox")
    kwargs.setdefault("bbox_margin", 0.30)
    kwargs.setdefault("return_index", True)

    # Important: use cached 64x64 crops and skip online bbox crop.
    kwargs.setdefault("image_dir_name", "images_cropped_6464")
    kwargs.setdefault("already_cropped", True)

    return kwargs


class CelebACroppedTrainUnbiased(CelebA):
    def __init__(self, **kwargs):
        kwargs = _set_celeba_cropped_default_kwargs(kwargs)
        super().__init__(
            path=CELEBA_ROOT,
            indices_file=CELEBA_TRAIN_UNBIASED_SPLIT,
            **kwargs,
        )


class CelebACroppedTrainEyeglassesMaleHatBiased(CelebA):
    def __init__(self, **kwargs):
        kwargs = _set_celeba_cropped_default_kwargs(kwargs)
        super().__init__(
            path=CELEBA_ROOT,
            indices_file=CELEBA_TRAIN_EYEGLASSES_MALE_HAT_BIASED_SPLIT,
            **kwargs,
        )


class CelebACroppedTestFixed(CelebA):
    def __init__(self, **kwargs):
        kwargs = _set_celeba_cropped_default_kwargs(kwargs)
        super().__init__(
            path=CELEBA_ROOT,
            indices_file=CELEBA_TEST_FIXED_SPLIT,
            **kwargs,
        )