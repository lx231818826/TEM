import numpy, gzip
import cPickle as pickle


filename = "results/log_baseline_15000_test_all.pkl.gz"
filename = "results/sae_baseline_15000_test_all.pkl.gz"

filename = "results/sae_tl_50000_15000_test_all.pkl.gz"
filename = "results/sae_tl_30000_15000_test_all.pkl.gz"
filename = "results/sae_tl_20000_15000_test_all.pkl.gz"


res = pickle.load(gzip.open(filename))
prec = res[:,0]
rec = res[:,1]

numpy.mean(2 * prec * rec / (prec+rec))*100
numpy.std(2 * prec * rec / (prec+rec))*100