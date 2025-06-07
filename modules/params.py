from dataclasses import dataclass

@dataclass
class ModelParams:
    model_type: str
    Ka: int
    La: int
    Kb: int
    Lb: int
    Mb: int
    Kc: int
    Lc: int
    Mc: int
    epochs: int
    lr: float

