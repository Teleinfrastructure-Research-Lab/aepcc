import open3d as o3d
from collections import OrderedDict

class MeshCache:
    def __init__(self, max_cache_size=100):
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()  # Maintains order for LRU behavior

    def load_mesh(self, path):
        """Load mesh with caching and LRU replacement."""
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]

        mesh = o3d.io.read_triangle_mesh(path)

        self.cache[path] = mesh

        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)

        return mesh