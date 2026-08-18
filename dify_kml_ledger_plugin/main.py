from dify_plugin import DifyPluginEnv, Plugin

from tools.kml_to_ledger import KmlToLedgerTool


plugin = Plugin(DifyPluginEnv(MAX_REQUEST_TIMEOUT=120))


if __name__ == "__main__":
    plugin.run()
