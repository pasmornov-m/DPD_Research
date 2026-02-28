import torch
import os


class BaseModel:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.class_name = self.__class__.__name__
        self._filename = None
    
    def count_params(self) -> int:
        total = 0

        for p in self.parameters():
            if not p.requires_grad:
                continue

            n = p.numel()

            if p.is_complex():
                total += 2 * n
            else:
                total += n

        return total
    
    @property
    def filename(self):
        if self._filename is None:
            self._filename = self._get_filename()
        return self._filename
    
    def _get_filename(self):
        raise NotImplementedError
    
    def save_weights(self, directory: str = "model_params"):
        os.makedirs(directory, exist_ok=True)
        full_path = f"{directory}/{self.filename}"
        torch.save(self.state_dict(), full_path)
        print(f"Model weights saved to {full_path}")

    def load_weights(self, directory: str = "model_params") -> bool:
        full_path = f"{directory}/{self.filename}"
        if os.path.isfile(full_path):
            state_dict = torch.load(full_path, map_location="cpu")
            self.load_state_dict(state_dict)
            print(f"Model weights loaded from {full_path}")
            return True
        else:
            print(f"No saved weights found at {full_path}, initializing new parameters.")
            return False