from strictyaml import Map, Bool, Str, Int, Float


SCHEMA = Map({
    "experiment": Map({
        "alpha": Float(),
        "delta": Float(),
        "splits": Int(),
        "ratio": Float(),
        "norm": Str(),
        "n_s": Int(),
        "batch_size": Int(),
        "dataset": Str(),
        "arc": Str(),
        "my_model": Bool(),
        "n_test": Int(),
    }),
    "path": Map({
        "root": Str(),
        "datasets": Str(),
        "checkpoints": Str(),
        "adversarial_examples": Str(),
        "bounds": Str(),
        "results": Str(),
    }),
    "rand": Map({
        "seed": Int(),
    }),
})
