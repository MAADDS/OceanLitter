class Evaluator:
    def __init__(self, image_path, txt_path):
        self.image_path = image_path
        self.txt_path = txt_path

    def evaluate(self):
        print(f"{self.image_path}, {self.txt_path}")
