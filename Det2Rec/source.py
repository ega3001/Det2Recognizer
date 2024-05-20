import io

import PIL.Image
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog


class Det2Recognizer():
    def __init__(self, cfg_path, model_path: str = '', device: str = "cpu", classes: list = []):
        # create config
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.DEVICE = device
        if model_path:
            cfg.MODEL.WEIGHTS = model_path

        self._classes = MetadataCatalog.get(
            cfg.DATASETS.TRAIN[0]
        ).thing_classes
        self._filterClasses = classes
        self._predictor = DefaultPredictor(cfg)

    def _prepare_result(self, instances):
        fields = instances.get_fields()
        scores = fields["scores"].tolist()
        pclasses = fields["pred_classes"].tolist()
        pboxes = fields["pred_boxes"].tensor.tolist()
        assert len(scores) == len(pclasses) == len(pboxes)
        result = []
        for i, pclass in enumerate(pclasses):
            if self._filterClasses and self._classes[pclass] not in self._filterClasses:
                continue

            obj = {
                self._classes[pclass]: {
                    "percent": scores[i],
                    "bounds": pboxes[i]
                }
            }
            result.append(obj)
        return result

    def score(self, image_buffer: bytes):
        image = PIL.Image.open(io.BytesIO(image_buffer)).convert("RGB")
        open_cv_image = np.array(image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        # make prediction
        result = self._predictor(open_cv_image)

        return self._prepare_result(result["instances"])
