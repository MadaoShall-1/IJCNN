"""IJCNN-Qiwei retained reasoning architecture.

The package initializer intentionally avoids importing training modules. Those
modules depend on torch and sentence-transformers, while the EXACT API layer
must remain lightweight enough to start in a serving environment.
"""

__all__: list[str] = []
