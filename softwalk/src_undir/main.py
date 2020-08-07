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

	parser.add_argument('--input', nargs='?', default='graph/bench1648.edgelist',
	                    help='Input graph path')


	parser.add_argument('--output', nargs='?', default='emb_undir/bench1648.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=35,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 100.')

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

	parser.add_argument('--a_x', type=float, default=2,
	                    help='Decreasing index hyperparameter. Default is 2.')
# a_x=1.5, 2, 4, 6, 8, 10
	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=True)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')

	parser.set_defaults(directed=False)

	return parser.parse_args()



def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	print('weight: ', args.weighted)
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
		for edge in G.edges():
			if G[edge[0]][edge[1]]['weight'] == 0:
				G[edge[0]][edge[1]]['weight'] = 10**(-6)
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
	G = softwalk.Graph(nx_G, args.directed, args.p, args.q, args.a_x)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
	print('The program end!')
