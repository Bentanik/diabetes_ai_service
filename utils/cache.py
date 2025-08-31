import time
from collections import OrderedDict

class LRUCache:
    def __init__(self, maxsize=500, ttl=3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()  # key: (value, timestamp)

    def __contains__(self, key):
        if key not in self.cache:
            return False
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return False
        return True

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        value, _ = self.cache[key]
        # Đưa lên đầu (MRU)
        # MRU: Most Recently Used
        self.cache.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        # Xóa nếu đã tồn tại
        if key in self.cache:
            self.cache.pop(key)
        # Kiểm tra giới hạn
        elif len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)  # xóa LRU
        # Thêm mới với timestamp
        self.cache[key] = (value, time.time())

    def size(self):
        # Dọn dẹp các item hết hạn
        now = time.time()
        expired = [k for k, (_, ts) in self.cache.items() if now - ts > self.ttl]
        for k in expired:
            del self.cache[k]
        return len(self.cache)