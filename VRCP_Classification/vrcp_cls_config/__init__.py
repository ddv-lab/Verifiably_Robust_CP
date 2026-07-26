from .config import Config, CFG_EXPERIMENT, CFG_PATH, CFG_RAND

__all__ = ["Config", "CFG_EXPERIMENT", "CFG_PATH", "CFG_RAND"]

# Unlike VRCP_Regression's config (which auto-loads on import), this module
# does NOT auto-load: callers must run `Config.load(cli_overrides=vars(args))`
# after parsing CLI args, since CLI flags are allowed to override config.yaml
# values (needed for parameter sweeps, e.g. `--n_s 512`). Access `Config.DATA`
# after calling `Config.load(...)`.
