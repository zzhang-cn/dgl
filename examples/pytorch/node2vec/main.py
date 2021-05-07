import time
from dgl.sampling.node2vec import node2vec_randomwalk
from model import Node2vecModel
from utils import load_graph, parse_arguments


def time_randomwalk(graph, args):
    """
    Test cost time of random walk
    """

    start_time = time.time()

    # default setting for testing
    params = {'p': 0.25,
              'q': 4,
              'walk_length': 50}

    for i in range(args.runs):
        node2vec_randomwalk(graph, graph.nodes(), **params)
    end_time = time.time()
    cost_time_avg = (end_time-start_time)/args.runs
    print("Run dataset {} {} trails, mean cost time: {:.3f}s".format(args.dataset, args.runs, cost_time_avg))


def train_node2vec(graph, eval_set, args):
    """
    Train node2vec model
    """
    trainer = Node2vecModel(graph,
                            embedding_dim=args.embedding_dim,
                            walk_length=args.walk_length,
                            p=args.p,
                            q=args.q,
                            num_walks=args.num_walks,
                            eval_set=eval_set,
                            eval_steps=1,
                            device=args.device)

    trainer.train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.01)


if __name__ == '__main__':

    args = parse_arguments()
    graph, eval_set = load_graph(args.dataset)

    if args.task == 'train':
        print("Perform training node2vec model")
        train_node2vec(graph, eval_set, args)
    elif args.task == 'test':
        print("Perform testing random walk")
        time_randomwalk(graph, args)
    else:
        raise ValueError('Task type error!')
