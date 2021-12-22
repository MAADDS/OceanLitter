from ImageStitcher import ImageStitcher
from LitterDetector import LitterDetector
from Evaluator import Evaluator
from pathlib import Path

""" Initialize folders """
INPUT_PATH = "./input"
STITCHED_PATH = "./stitched"
DETECT_RESULT_PATH = "./detection_result"
EVAL_PATH = "./eval"

input_dir = Path(INPUT_PATH)
stitched_dir = Path(STITCHED_PATH)
detect_result_dir = Path(DETECT_RESULT_PATH)
eval_dir = Path(EVAL_PATH)

Path(input_dir).mkdir(parents=True, exist_ok=True)
Path(stitched_dir).mkdir(parents=True, exist_ok=True)
Path(detect_result_dir).mkdir(parents=True, exist_ok=True)
Path(eval_dir).mkdir(parents=True, exist_ok=True)


""" Stitch! """
print([str(x) for x in input_dir.iterdir() if x.is_dir()])  # subdir of input_dir

[ImageStitcher(str(x)).stitch() for x in input_dir.iterdir() if x.is_dir()]


# """ Litter Detection """
[LitterDetector(str(x)).detect() for x in stitched_dir.glob("*.jpg")]

""" Evaluate Pollution Level """
for image, txt in zip(stitched_dir.glob("*.jpg"), detect_result_dir.glob("*.txt")):
    Evaluator(str(image), str(txt)).evaluate()
