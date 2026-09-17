"""Minimal numpy loader for torch .pth files (zip format, no torch needed)."""
import zipfile, pickle, struct, numpy as np, io

DTYPES = {
    'FloatStorage': np.float32, 'DoubleStorage': np.float64,
    'HalfStorage': np.float16, 'LongStorage': np.int64,
    'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8,
    'BoolStorage': np.bool_, 'ComplexFloatStorage': np.complex64,
    'ComplexDoubleStorage': np.complex128,
}

class _Unpickler(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix
    def find_class(self, mod, name):
        if mod.startswith('torch') and name in DTYPES:
            return name                      # marker string
        if mod == 'torch._utils' and name in ('_rebuild_tensor_v2','_rebuild_tensor'):
            return _rebuild
        if mod == 'collections' and name == 'OrderedDict':
            import collections; return collections.OrderedDict
        if mod == 'torch' and name == 'Size':
            return tuple
        return super().find_class(mod, name)
    def persistent_load(self, pid):
        # pid = ('storage', StorageType, key, location, numel)
        _, stype, key, _loc, numel = pid
        dt = DTYPES[stype] if isinstance(stype, str) else np.float32
        for cand in (f'{self.prefix}/data/{key}', f'{self.prefix}/{key}'):
            try:
                raw = self.zf.read(cand); break
            except KeyError:
                continue
        else:
            raise KeyError(key)
        return np.frombuffer(raw, dtype=dt).copy()

def _rebuild(storage, offset, size, stride, *a, **k):
    size = tuple(size); stride = tuple(stride)
    n = int(np.prod(size)) if size else 1
    flat = storage[offset:offset+n] if _contig(size,stride) else None
    if flat is not None:
        return flat.reshape(size)
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=size,
        strides=tuple(s*storage.itemsize for s in stride)).copy()

def _contig(size, stride):
    exp, acc = [], 1
    for s in reversed(size):
        exp.append(acc); acc *= s
    return tuple(reversed(exp)) == tuple(stride)

def load(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    prefix = names[0].split('/')[0]
    data = zf.read(f'{prefix}/data.pkl')
    return _Unpickler(io.BytesIO(data), zf, prefix).load()
