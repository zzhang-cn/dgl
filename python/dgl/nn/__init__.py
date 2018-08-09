import os
__backend__ = os.environ.get('DGLBACKEND', 'pytorch').lower()

if __backend__ == 'numpy':
    pass
elif __backend__ == 'pytorch':
    from .pytorch import *
else:
    raise Exception("Unsupported backend %s" % __backend__)
