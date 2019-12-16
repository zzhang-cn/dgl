"""Atomic Convolutional Networks for Predicting Protein-Ligand Binding Affinity"""
import itertools
import numpy as np
import torch
import torch.nn as nn

from ... import backend
from ...nn.pytorch import AtomicConv

def truncated_normal_(tensor, mean=0., std=1.):
    """Fills the given tensor in-place with elements sampled from the truncated normal
    distribution parameterized by mean and std.

    The generated values follow a normal distribution with specified mean and
    standard deviation, except that values whose magnitude is more than 2 std
    from the mean are dropped.

    We credit to Ruotian Luo for this implementation:
    https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15.

    Parameters
    ----------
    tensor : Float32 tensor of arbitrary shape
        Tensor to be filled.
    mean : float
        Mean of the truncated normal distribution.
    std : float
        Standard deviation of the truncated normal distribution.
    """
    shape = tensor.shape
    tmp = tensor.new_empty(shape + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class ACNNPredictor(nn.Module):
    """Predictor for ACNN.

    Parameters
    ----------
    in_size : int
        Number of radial filters used.
    hidden_sizes : list of int
        Specifying the hidden sizes for all layers in the predictor.
    weight_init_stddevs : list of float
        Specifying the standard deviations to use for truncated normal
        distributions in initialzing weights for the predictor.
    dropouts : list of float
        Specifying the dropouts to use for all layers in the predictor.
    features_to_use : None or float tensor of shape (T)
        In the original paper, these are atomic numbers to consider, representing the types
        of atoms. T for the number of types of atomic numbers. Default to None.
    num_tasks : int
        Output size.
    """
    def __init__(self, in_size, hidden_sizes, weight_init_stddevs,
                 dropouts, features_to_use, num_tasks):
        super(ACNNPredictor, self).__init__()

        if type(features_to_use) != type(None):
            in_size *= len(features_to_use)

        modules = []
        for i, h in enumerate(hidden_sizes):
            linear_layer = nn.Linear(in_size, h)
            truncated_normal_(linear_layer.weight, std=weight_init_stddevs[i])
            modules.append(linear_layer)
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropouts[i]))
            in_size = h
        linear_layer = nn.Linear(in_size, num_tasks)
        truncated_normal_(linear_layer.weight, std=weight_init_stddevs[-1])
        modules.append(linear_layer)
        self.project = nn.Sequential(*modules)

    @staticmethod
    def sum_nodes(batch_size, batch_num_nodes, feats):
        """Segment sum for node features.

        Parameters
        ----------
        batch_size : int
            Number of graphs in a batch.
        batch_num_nodes : list of int
            The ``i``th element represents the number of nodes in the ``i``th graph.
        feats : Float32 Tensor of shape (V, O)
            Updated node features. O for the number of tasks.

        Returns
        -------
        Float32 Tensor of shape (batch_size, O)
            Computed graph representations. O for the number of tasks.
        """
        seg_id = torch.from_numpy(np.arange(batch_size, dtype='int64').repeat(batch_num_nodes))
        seg_id = seg_id.to(feats.device)

        return backend.unsorted_1d_segment_sum(feats, seg_id, batch_size, 0)

    def forward(self, ligand_graph, protein_graph, complex_graph,
                ligand_conv_out, protein_conv_out, complex_conv_out):
        """Perform the prediction.

        Parameters
        ----------
        ligand_graph : DGLHeteroGraph
            DGLHeteroGraph for the ligand graph.
        protein_graph : DGLHeteroGraph
            DGLHeteroGraph for the protein graph.
        complex_graph : DGLHeteroGraph
            DGLHeteroGraph for the complex graph.
        ligand_conv_out : Float32 tensor of shape (V1, K * T)
            Updated ligand node representations. V1 for the number of
            atoms in the ligand, K for the number of radial filters,
            and T for the number of types of atomic numbers.
        protein_conv_out : Float32 tensor of shape (V2, K * T)
            Updated protein node representations. V2 for the number of
            atoms in the protein, K for the number of radial filters,
            and T for the number of types of atomic numbers.
        complex_conv_out : Float32 tensor of shape (V1 + V2, K * T)
            Updated complex node representations. V1 and V2 seprately
            for the number of atoms in the ligand and protein, K for
            the number of radial filters, and T for the number of
            types of atomic numbers.

        Returns
        -------
        Float32 tensor of shape (B, O)
            Predicted protein-ligand binding affinity. B for the number
            of protein-ligand pairs in the batch and O for the number of tasks.
        """
        ligand_feats = self.project(ligand_conv_out)   # (V1, O)
        protein_feats = self.project(protein_conv_out) # (V2, O)
        complex_feats = self.project(complex_conv_out) # (V1+V2, O)

        ligand_energy = self.sum_nodes(                # (B, O)
            ligand_graph.batch_size,
            ligand_graph.batch_num_nodes,
            ligand_feats)
        protein_energy = self.sum_nodes(               # (B, O)
            protein_graph.batch_size,
            protein_graph.batch_num_nodes,
            protein_feats)

        batch_num_nodes = {
            'ligand_atom': ligand_graph.batch_num_nodes,
            'protein_atom': protein_graph.batch_num_nodes
        }
        complex_energy_ = self.sum_nodes(              # (B, O)
            complex_graph.batch_size * 2,
            list(itertools.chain.from_iterable(
                [batch_num_nodes[nty] for nty in complex_graph.original_ntypes])),
            complex_feats)
        complex_energy = complex_energy_[:complex_graph.batch_size] + \
                         complex_energy_[complex_graph.batch_size:]

        return complex_energy - (ligand_energy + protein_energy)

class ACNN(nn.Module):
    """Atomic Convolutional Networks.

    The model was proposed in `Atomic Convolutional Networks for
    Predicting Protein-Ligand Binding Affinity <https://arxiv.org/abs/1703.10603>`__.

    Parameters
    ----------
    hidden_sizes : list of int
        Specifying the hidden sizes for all layers in the predictor.
    weight_init_stddevs : list of float
        Specifying the standard deviations to use for truncated normal
        distributions in initialzing weights for the predictor.
    dropouts : list of float
        Specifying the dropouts to use for all layers in the predictor.
    features_to_use : None or float tensor of shape (T)
        In the original paper, these are atomic numbers to consider, representing the types
        of atoms. T for the number of types of atomic numbers. Default to None.
    radial : None or list
        If not None, the list consists of 3 lists of floats, separately for the
        options of interaction cutoff, the options of rbf kernel mean and the
        options of rbf kernel scaling. If None, a default option of
        ``[[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]]`` will be used.
    num_tasks : int
        Number of output tasks.
    """
    def __init__(self, hidden_sizes, weight_init_stddevs, dropouts,
                 features_to_use=None, radial=None, num_tasks=1):
        super(ACNN, self).__init__()

        if radial is None:
            radial = [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]]
        # Take the product of sets of options and get a list of 3-tuples.
        radial_params = [x for x in itertools.product(*radial)]
        radial_params = torch.stack(list(map(torch.tensor, zip(*radial_params))), dim=1)

        interaction_cutoffs = radial_params[:, 0]
        rbf_kernel_means = radial_params[:, 1]
        rbf_kernel_scaling = radial_params[:, 2]

        self.ligand_conv = AtomicConv(interaction_cutoffs, rbf_kernel_means,
                                      rbf_kernel_scaling, features_to_use)
        self.protein_conv = AtomicConv(interaction_cutoffs, rbf_kernel_means,
                                       rbf_kernel_scaling, features_to_use)
        self.complex_conv = AtomicConv(interaction_cutoffs, rbf_kernel_means,
                                       rbf_kernel_scaling, features_to_use)
        self.predictor = ACNNPredictor(radial_params.shape[0], hidden_sizes,
                                       weight_init_stddevs, dropouts, features_to_use, num_tasks)

    def forward(self, graph):
        """Apply the model for prediction.

        Parameters
        ----------
        graph : DGLHeteroGraph
            DGLHeteroGraph consisting of the ligand graph, the protein graph
            and the complex graph, along with preprocessed features.

        Returns
        -------
        Float32 tensor of shape (B, O)
            Predicted protein-ligand binding affinity. B for the number
            of protein-ligand pairs in the batch and O for the number of tasks.
        """
        ligand_graph = graph[('ligand_atom', 'ligand', 'ligand_atom')]
        # Todo (Mufei): remove the two lines below after better built-in support
        ligand_graph.batch_size = graph.batch_size
        ligand_graph.batch_num_nodes = graph.batch_num_nodes('ligand_atom')

        ligand_graph_node_feats = ligand_graph.ndata['atomic_number']
        assert ligand_graph_node_feats.shape[-1] == 1
        ligand_graph_distances = ligand_graph.edata['distance']
        ligand_conv_out = self.ligand_conv(ligand_graph,
                                           ligand_graph_node_feats,
                                           ligand_graph_distances)

        protein_graph = graph[('protein_atom', 'protein', 'protein_atom')]
        # Todo (Mufei): remove the two lines below after better built-in support
        protein_graph.batch_size = graph.batch_size
        protein_graph.batch_num_nodes = graph.batch_num_nodes('protein_atom')

        protein_graph_node_feats = protein_graph.ndata['atomic_number']
        assert protein_graph_node_feats.shape[-1] == 1
        protein_graph_distances = protein_graph.edata['distance']
        protein_conv_out = self.protein_conv(protein_graph,
                                             protein_graph_node_feats,
                                             protein_graph_distances)

        complex_graph = graph[:, 'complex', :]
        # Todo (Mufei): remove the four lines below after better built-in support
        complex_graph.batch_size = graph.batch_size
        complex_graph.original_ntypes = graph.ntypes
        complex_graph.batch_num_protein_nodes = graph.batch_num_nodes('protein_atom')
        complex_graph.batch_num_ligand_nodes = graph.batch_num_nodes('ligand_atom')

        complex_graph_node_feats = complex_graph.ndata['atomic_number']
        assert complex_graph_node_feats.shape[-1] == 1
        complex_graph_distances = complex_graph.edata['distance']
        complex_conv_out = self.complex_conv(complex_graph,
                                             complex_graph_node_feats,
                                             complex_graph_distances)

        return self.predictor(
            ligand_graph, protein_graph, complex_graph,
            ligand_conv_out, protein_conv_out, complex_conv_out)
