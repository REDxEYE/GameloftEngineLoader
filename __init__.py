def pig_init():
    pass


def plugin_init():
    pass


def pig_load():
    pass


plugin_info = {
    "name": "Gameloft engine",
    "id": "GameloftEngineLoader",
    "description": "Import Gameloft engine assets",
    "version": (0, 1, 0),
    "loaders": [
        {
            "name": "Load .pig file",
            "id": "gle_pig",
            "exts": ("*.pig",),
            "init_fn": pig_init,
            "import_fn": pig_load,
            "properties": [
            ]
        },
    ],
    "init_fn": plugin_init
}
