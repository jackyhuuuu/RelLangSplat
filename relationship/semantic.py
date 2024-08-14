from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


class SemanticResolver(object):
    def __init__(self) -> None:
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate_semantic_relationship(self, image: Image, query: str) -> bool:
        text = self._semantic_relationship_prompt(query)
        inputs = self.processor(images=image,  text=text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=50)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    @staticmethod
    def _semantic_relationship_prompt(query: str):
        return f"Question: Is the '''{query}''' true? Answer: "
