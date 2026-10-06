"""Paper conditions; configurations vary stages, never the metric definitions."""
from dataclasses import asdict, dataclass
from pathlib import Path
from .io import read_json

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/paper.json"


@dataclass(frozen=True)
class Condition:
    name: str
    dictionary: bool = True
    csc: bool = True
    review: bool = True
    mask: str = "full"
    margin: float = 0.70
    generation: str | None = None

    @property
    def stochastic(self):
        return self.review or self.generation is not None


CONDITIONS = {
    "care": Condition("care"),
    "identity": Condition("identity", False, False, False, "none"),
    "a_b": Condition("a_b", review=False),
    "a": Condition("a", csc=False, review=False),
    "a_c": Condition("a_c", csc=False),
    "b_c": Condition("b_c", dictionary=False, mask="regex_only"),
    "no_mask": Condition("no_mask", mask="none"),
    "delta_zero": Condition("delta_zero", margin=0.0),
    **{name: Condition(name, False, False, False, "none", generation=name)
       for name in ["free", "conservative", "dictionary"]},
    **{f"delta_{round(d * 100):03d}": Condition(f"delta_{round(d * 100):03d}", margin=d)
       for d in [0.4, 0.5, 0.6, 0.65, 0.75, 0.8, 0.9]},
}


def load_settings(path):
    value = read_json(path)
    if value["tensor_parallel_size"] != 1:
        raise ValueError("This release targets one GPU")
    if value["context_window"] != 25 or value["relm_phonetic_filter"] != "off":
        raise ValueError("The paper uses a 25-character context and no phonetic CSC filter")
    if value["temperature"] != 0 or value["enable_thinking"]:
        raise ValueError("The paper uses temperature=0 and enable_thinking=false")
    if value["repeats"] < 1 or len(set(value["files"])) != len(value["files"]) or not value["files"]:
        raise ValueError("Expected unique file IDs and at least one repetition")
    return value


def select_conditions(config, suite, names=None):
    names = names or read_json(Path(config).parent / f"{suite}.json")["conditions"]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate conditions")
    return [asdict(CONDITIONS[name]) for name in names]
