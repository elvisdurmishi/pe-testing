import json
import numpy as np
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

@dataclass
class AIModel:
    model_path: Path
    tokenizer_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    model = None
    tokenizer = None
    metadata = None
    labels_legend = None

    def __post_init__(self):
        if self.model_path.exists():
            self.model = load_model(self.model_path)
        if self.tokenizer_path:
            if self.tokenizer_path.exists():
                if not self.tokenizer_path.name.endswith("json"):
                    raise Exception("Tokenizer should be in json format")
                self.tokenizer = tokenizer_from_json(self.tokenizer_path.read_text())
        if self.metadata_path:
            if self.metadata_path.exists():
                if not self.metadata_path.name.endswith("json"):
                    raise Exception("Metadata should be in json format")
                self.metadata = json.loads(self.metadata_path.read_text())

    def get_model(self):
        if not self.model:
            raise Exception("Model not implemented")
        return self.model

    def get_tokenizer(self):
        if not self.tokenizer:
            raise Exception("Tokenizer not implemented")
        return self.tokenizer

    def get_metadata(self):
        if not self.metadata:
            raise Exception("Metadata not implemented")
        return self.metadata

    def get_labels_legend(self):
        metadata = self.get_metadata()
        if not metadata:
            raise Exception("Metadata not implemented, to get labels legends")
        if metadata.get('labels_legend_inverted') == None:
            raise Exception("labels_legend_inverted key is missing from metadata")
        if len(metadata.get('labels_legend_inverted').keys()) != 2:
            raise Exception("Labels legends is incorrect (less then 2 keys)")
        return metadata.get('labels_legend_inverted')

    def get_sequences_from_text(self, texts: List[str] ):
        tokenizer = self.get_tokenizer()
        return tokenizer.texts_to_sequences(texts)

    def get_input_from_sequences(self, sequences):
        max_sequence = self.get_metadata().get('max_sequence') or 1000
        return pad_sequences(sequences, maxlen=max_sequence)
         
    def get_label_prediction(self, index, value):
        legend = self.get_labels_legend()
        return {"label": legend[str(index)], "confidence": float(value)}
         
    def get_top_labeled_prediction(self, predictions):
        top_prediction_index = np.argmax(predictions)
        return self.get_label_prediction(top_prediction_index, predictions[top_prediction_index])

    def predict_text(self, body: str, include_top_prediction=True):
        model = self.get_model()
        sequences = self.get_sequences_from_text([body])
        x_input = self.get_input_from_sequences(sequences)
        predictions = model.predict(x_input)[0]
        labeled_predictions = [self.get_label_prediction(i, x) for i, x in enumerate(list(predictions))]
        results = {
            "predictions": labeled_predictions
        }

        if include_top_prediction:
            results["top_prediction"] = self.get_top_labeled_prediction(predictions)
        return results