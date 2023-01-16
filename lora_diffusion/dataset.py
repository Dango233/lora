import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import torch

OBJECT_TEMPLATE = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

STYLE_TEMPLATE = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


def _randomset(lis):
    ret = []
    for i in range(len(lis)):
        if random.random() < 0.5:
            ret.append(lis[i])
    return ret


def _shuffle(lis):

    return random.sample(lis, len(lis))


class PivotalTuningDatasetCapation(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    # TODO: cache the class latent
    def __init__(
        self,
        instance_data_root,
        stochastic_attribute,
        tokenizer,
        token_map: Optional[dict] = None,
        use_template: Optional[str] = None,
        class_data_root=None,
        class_prompt=None,
        size=512,
        h_flip=True,
        color_jitter=False,
        resize=True,
        use_face_segmentation_condition=False,
        blur_amount: int = 70,
        repeats = 12,
        use_preprocessed_mask = False,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.resize = resize

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        # Only process the images with the extension of .jpg, .jpeg, .png
        self.instance_images_path = [
            path
            for path in self.instance_images_path
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        self.num_instance_images = len(self.instance_images_path)
        self.token_map = token_map

        self.use_template = use_template
        self.templates = OBJECT_TEMPLATE if use_template == "object" else STYLE_TEMPLATE

        self._length = self.num_instance_images * repeats
        self.pointer = list(range(self.num_instance_images))
        self.randomized = False
        self.instance_latent_cached = False

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images * repeats)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None
        self.h_flip = h_flip
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                )
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.ColorJitter(0.1, 0.1)
                if color_jitter
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.use_face_segmentation_condition = use_face_segmentation_condition
        self.use_preprocessed_mask = use_preprocessed_mask
        if self.use_face_segmentation_condition:
            import mediapipe as mp

            mp_face_detection = mp.solutions.face_detection
            self.face_detection = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        self.blur_amount = blur_amount

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        i = index % self.num_instance_images
        if i == 0 and self.randomized >=1:
            random.shuffle(self.pointer)
        i = self.pointer[i]
        if self.instance_latent_cached:
            example["instance_latents"] = self.cached_instance_latent[i]
            if self.use_face_segmentation_condition:
                example["mask"] = self.cached_instance_mask[i]
        if not self.instance_latent_cached:
            instance_image = Image.open(
                self.instance_images_path[i]
            )
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            example["instance_images"] = self.image_transforms(instance_image)

        if self.use_template:
            assert self.token_map is not None
            input_tok = list(self.token_map.values())[0]

            text = random.choice(self.templates).format(input_tok)
        else:
            text = self.instance_images_path[i].stem
            if self.token_map is not None:
                for token, value in self.token_map.items():
                    text = text.replace(token, value)

        # print(text)
        if not self.instance_latent_cached:
            if self.use_preprocessed_mask:
                #insert /mask into the path
                mask_path = self.instance_images_path[i].parent / "masks" / self.instance_images_path[i].name
                mask_path = mask_path.parent / (mask_path.name + '.pt')
                mask = torch.load(mask_path)
                example["preprocessed_mask"] = mask
            elif self.use_face_segmentation_condition:
                image = cv2.imread(
                    str(self.instance_images_path[i])
                )
                results = self.face_detection.process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                )
                black_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

                if results.detections:

                    for detection in results.detections:

                        x_min = int(
                            detection.location_data.relative_bounding_box.xmin
                            * image.shape[1]
                        )
                        y_min = int(
                            detection.location_data.relative_bounding_box.ymin
                            * image.shape[0]
                        )
                        width = int(
                            detection.location_data.relative_bounding_box.width
                            * image.shape[1]
                        )
                        height = int(
                            detection.location_data.relative_bounding_box.height
                            * image.shape[0]
                        )

                        # draw the colored rectangle
                        black_image[y_min : y_min + height, x_min : x_min + width] = 255

                # blur the image
                black_image = Image.fromarray(black_image, mode="L").filter(
                    ImageFilter.GaussianBlur(radius=self.blur_amount)
                )
                # to tensor
                black_image = transforms.ToTensor()(black_image)
                # resize as the instance image
                black_image = transforms.Resize(
                    self.size, interpolation=transforms.InterpolationMode.BILINEAR
                )(black_image)

                example["mask"] = black_image

            if self.h_flip and random.random() > 0.5:
                hflip = transforms.RandomHorizontalFlip(p=1)

                example["instance_images"] = hflip(example["instance_images"])
                if self.use_face_segmentation_condition:
                    example["mask"] = hflip(example["mask"])

        example["instance_prompt_ids"] = self.tokenizer(
            text,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[i]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example
