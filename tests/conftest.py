"""Every test starts without plug-ins - never the ones in the user's plug-in folder."""

import pytest

from bictool.application import plugin_manager


@pytest.fixture(autouse=True)
def no_plugins():
    plugin_manager.use(plugin_manager.PluginManager())
    yield
    plugin_manager.use(None)
