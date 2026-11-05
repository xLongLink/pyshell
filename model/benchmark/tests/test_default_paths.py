from pathlib import Path
from model.benchmark.src.corpus import default_training_data


class TestDefaultTrainingData:
    def test_defaults_are_repository_relative_in_both_modes(self, subtests):
        repo = Path(__file__).resolve().parents[2]
        for serialization, directory in (("raw", "train/data/raw"), ("chat", "train/data/chat")):
            with subtests.test(serialization=serialization):
                assert default_training_data(serialization) == repo / directory
