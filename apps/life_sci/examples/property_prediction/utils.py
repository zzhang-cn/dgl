import dgl
import numpy as np
import random
import torch

from dglls.utils.featurizers import one_hot_encoding
from dglls.utils.mol_to_graph import smiles_to_bigraph
from dglls.utils.splitters import RandomSplitter

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_dataset_for_classification(args):
    """Load dataset for classification tasks.
    Parameters
    ----------
    args : dict
        Configurations.
    Returns
    -------
    dataset
        The whole dataset.
    train_set
        Subset for training.
    val_set
        Subset for validation.
    test_set
        Subset for test.
    """
    assert args['dataset'] in ['Tox21']
    if args['dataset'] == 'Tox21':
        from dglls.data import Tox21
        dataset = Tox21(smiles_to_bigraph, args['atom_featurizer'])
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])

    return dataset, train_set, val_set, test_set

def load_dataset_for_regression(args):
    """Load dataset for regression tasks.
    Parameters
    ----------
    args : dict
        Configurations.
    Returns
    -------
    train_set
        Subset for training.
    val_set
        Subset for validation.
    test_set
        Subset for test.
    """
    assert args['dataset'] in ['Alchemy', 'Aromaticity']

    if args['dataset'] == 'Alchemy':
        from dglls.data import TencentAlchemyDataset
        train_set = TencentAlchemyDataset(mode='dev')
        val_set = TencentAlchemyDataset(mode='valid')
        test_set = None

    if args['dataset'] == 'Aromaticity':
        from dglls.data import PubChemBioAssayAromaticity
        dataset = PubChemBioAssayAromaticity(smiles_to_bigraph,
                                             args['atom_featurizer'],
                                             args['bond_featurizer'])
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])

    return train_set, val_set, test_set

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    smiles : list
        List of smiles
    bg : BatchedDGLGraph
        Batched DGLGraphs
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks

def load_model(args):
    if args['model'] == 'GCN':
        from dglls.model import GCNPredictor
        model = GCNPredictor(in_feats=args['in_feats'],
                             hidden_feats=args['gcn_hidden_feats'],
                             classifier_hidden_feats=args['classifier_hidden_feats'],
                             n_tasks=args['n_tasks'])

    if args['model'] == 'GAT':
        from dglls.model import GATPredictor
        model = GATPredictor(in_feats=args['in_feats'],
                             hidden_feats=args['gat_hidden_feats'],
                             num_heads=args['num_heads'],
                             classifier_hidden_feats=args['classifier_hidden_feats'],
                             n_tasks=args['n_tasks'])

    if args['model'] == 'AttentiveFP':
        from dglls.model import AttentiveFPPredictor
        model = AttentiveFPPredictor(node_feat_size=args['node_feat_size'],
                                     edge_feat_size=args['edge_feat_size'],
                                     num_layers=args['num_layers'],
                                     num_timesteps=args['num_timesteps'],
                                     graph_feat_size=args['graph_feat_size'],
                                     n_tasks=args['n_tasks'],
                                     dropout=args['dropout'])

    if args['model'] == 'SchNet':
        from dglls.model import SchNetPredictor
        model = SchNetPredictor(node_feats=args['node_feats'],
                                hidden_feats=args['hidden_feats'],
                                classifier_hidden_feats=args['classifier_hidden_feats'],
                                n_tasks=args['n_tasks'])

    if args['model'] == 'MGCN':
        from dglls.model import MGCNPredictor
        model = MGCNPredictor(feats=args['feats'],
                              n_layers=args['n_layers'],
                              classifier_hidden_feats=args['classifier_hidden_feats'],
                              n_tasks=args['n_tasks'])

    return model

def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]
