from .config import Config, CFG_GLOBAL, CFG_CP, CFG_CQR, CFG_DATA, CFG_DEBUG, CFG_ENV, CFG_PATH, CFG_RAND, CFG_SIM

if not Config.CONFIG_LOADED:
    Config.load()
    CONFIG = Config.DATA
