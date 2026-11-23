"""
Plug-in registries.

Every swappable component (landmark extractor, transform, model) registers
itself under a short name. Configs refer to components by that name, so adding
a new model or preprocessing step never requires touching the training code:

    @MODELS.register("my_model")
    class MyModel(nn.Module): ...

    # configs/model/my_model.yaml
    name: my_model
    params: {...}
"""


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        def decorator(obj):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj
        return decorator

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def build(self, name: str, **params):
        return self.get(name)(**params)

    def names(self) -> list[str]:
        return sorted(self._items)


EXTRACTORS = Registry("extractor")
TRANSFORMS = Registry("transform")
MODELS = Registry("model")
