import dgl
import dgl.ndarray as nd
from dgl.utils import toindex
import numpy as np
import backend as F
import unittest

@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow" or dgl.backend.backend_name == "jax",
    reason="TF and JAX don't support inplace update"
)
def test_dlpack():
    # test dlpack conversion.
    def nd2th():
        ans = np.array([[1., 1., 1., 1.],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.]])
        x = nd.array(np.zeros((3, 4), dtype=np.float32))
        dl = x.to_dlpack()
        y = F.zerocopy_from_dlpack(dl)
        y[0] = 1
        print(x)
        print(y)
        assert np.allclose(x.asnumpy(), ans)

    def th2nd():
        ans = np.array([[1., 1., 1., 1.],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.]])
        x = F.zeros((3, 4))
        dl = F.zerocopy_to_dlpack(x)
        y = nd.from_dlpack(dl)
        x[0] = 1
        print(x)
        print(y)
        assert np.allclose(y.asnumpy(), ans)

    def th2nd_incontiguous():
        x = F.astype(F.tensor([[0, 1], [2, 3]]), F.int64)
        ans = np.array([0, 2])
        y = x[:2, 0]
        # Uncomment this line and comment the one below to observe error
        #dl = dlpack.to_dlpack(y)
        dl = F.zerocopy_to_dlpack(y)
        z = nd.from_dlpack(dl)
        print(x)
        print(z)
        assert np.allclose(z.asnumpy(), ans)

    nd2th()
    th2nd()
    th2nd_incontiguous()
