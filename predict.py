# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from TTS.api import TTS


class Predictor(BasePredictor):
    def setup(self) -> None:

        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)

    def predict(
        self,
        text: str = Input(description="Text to synthesize"),
        language: str = Input(description="Languaage"),
        speaker_wav: Path = Input(description="Original speaker audio")
    ) -> Path:

        path = self.model.tts_to_file(text=text, 
        file_path = "output.wav",
        speaker_wav = speaker_wav,
        language= language
        )

        return Path(path)