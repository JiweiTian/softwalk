import argparse
import numpy as np
import networkx as nx
import softwalk
from gensim.models import Word2Vec
import scipy.io as sio

def parse_args():
	'''
	Parses the softwalk arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='Data/case25k_w3.edgelist',
	                    help='Input graph path')

	parser.add_argument('--input_dir', nargs='?', default='Data/case25k_dir.edgelist',
	                    help='Input graph path direction')

	parser.add_argument('--output', nargs='?', default='emb/case25k_w3.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=100,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=20,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=10, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=4,
	                    help='Number of parallel workers. Default is 4.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--a_x', type=float, default=5,
	                    help='Direction Degrade index')
	parser.add_argument('--b_x', type=float, default=2,
	                    help='Weight degrade index.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=True)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')

	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_direction(args):
	G_dir = nx.read_edgelist(args.input_dir, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	return G_dir

def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	print('weight: ', args.weighted)
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph(args)
	nx_G_dir = read_direction(args)
	G = softwalk.Graph(nx_G, nx_G_dir, args.directed, args.p, args.q, args.a_x, args.b_x)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
	print('The program end!')
