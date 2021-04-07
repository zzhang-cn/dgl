"""API creating NCCL communicators."""

class UniqueId(object):
    """ Class for allowing python code to create and communicate NCCL Unique
        IDs, needed for creating communicators.
    """
    def __init__(self, id_str=None):
        """ Create an object reference the current NCCL unique id.
        """

    def __str__(self):
        return "UniqueId"


class Communicator(object):
    """ High-level wrapper for NCCL communication.
    """
    def __init__(self, size, rank, unique_id):
        """ Create a new NCCL communicator.

            Parameters
            ----------
            size : int
                The number of processes in the communicator.
            rank : int
                The rank of the current process in the communicator.
            unique_id : NCCLUniqueId
                The unique id of the root process (rank=0).
        """

    def sparse_all_to_all_push(self, idx, value, mode):
        """ Perform an all-to-all-v operation, where by all processors send out
            a set of indices and corresponding values.

            Parameters
            ----------
            idx : IdArray
                The 1D set of indices to send to other processors.
            value : NDArray
                The multi-dimension set of values to send to other processors.
                The 0th dimension must match that of `idx`.
            mode : int
                The method for assigning indices to processors.

            Returns
            -------
            IdArray
                The set of recieved indices.
            NDArray
                The set of recieved values.

        """
        raise NotImplementedError

    def sparse_all_to_all_pull(self, req_idx, value, mode):
        """ Perform an all-to-all-v operation, where by all processors request
            the values corresponding to ther set of indices.

            Parameters
            ----------
            req_idx : IdArray
                The set of indices this processor is requesting.
            value : NDArray
                The multi-dimension set of values that can be requested from
                this processor.
            mode : int
                The method for assigning indices to processors.

            Returns
            -------
            NDArray
                The set of recieved values, corresponding to `req_idx`.

        """
        raise NotImplementedError

    def rank(self):
        """ Get the rank of this process in this communicator.

            Returns
            -------
            int
                The rank of this process.
        """
        raise NotImplementedError

    def size(self):
        """ Get the size of this communicator.

            Returns
            -------
            int
                The number of processes in this communicator.
        """
        raise NotImplementedError
