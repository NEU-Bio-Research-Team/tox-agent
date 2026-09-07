"""ToxAgent control plane.

Importing this package must stay cheap and dependency-free: no database
connection, no HTTP client, no runtime process. Composition happens in
``toxagent.api.app``.
"""

__version__ = "0.1.0.dev0"
