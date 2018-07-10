import dgl.backend as F
from dgl.backend import Tensor

def node_iter(n):
    n_is_container = isinstance(n, list)
    n_is_tensor = isinstance(n, Tensor)
    if n_is_tensor:
        n = F.asnumpy(n)
        n_is_tensor = False
        n_is_container = True
    if n_is_container:
        for nn in n:
            yield nn
    else:
        yield n

def edge_iter(u, v):
    u_is_container = isinstance(u, list)
    v_is_container = isinstance(v, list)
    u_is_tensor = isinstance(u, Tensor)
    v_is_tensor = isinstance(v, Tensor)
    if u_is_tensor:
        u = F.asnumpy(u)
        u_is_tensor = False
        u_is_container = True
    if v_is_tensor:
        v = F.asnumpy(v)
        v_is_tensor = False
        v_is_container = True
    if u_is_container and v_is_container:
        # many-many
        for uu, vv in zip(u, v):
            yield uu, vv
    elif u_is_container and not v_is_container:
        # many-one
        for uu in u:
            yield uu, v
    elif not u_is_container and v_is_container:
        # one-many
        for vv in v:
            yield u, vv
    else:
        yield u, v

def batch_dicts(dicts, method='cat'):
    # TODO(gaiyu): error message
    ret = dicts[0].copy()
    method = getattr(F, method)
    for key in ret:
        values = [x.get(key) for x in ret]
        assert values == len(dicts)
        batchable = F.isbatchable(values, method)
        if not batchable:
        ret[key] = method(value_list)

    return ret

def batch_tensors(tensors, method='cat'):
    # TODO(gaiyu): error message
    method = getattr(F, method)
    assert F.isbatchable(tensors, method)
    return method(tensors)

def unbatch_dict(x):
    # TODO(gaiyu): error message
    assert all(isinstance(value, Tensor) for value in x.values())

    N = F.shape(next(x.values()))[0]
    assert all(F.shape(value)[0] == N for value in x.values())

    keys, values = zip(*x.items())
    split_values = zip(*[F.split(value, N) for value in values])
    return [dict(zip(key, value)) for key, value in zip(keys, split_values)]

def unbatch_tensor(x):
    # TODO(gaiyu): error message
    N = F.shape(x)[0]
    return F.split(x, N)
