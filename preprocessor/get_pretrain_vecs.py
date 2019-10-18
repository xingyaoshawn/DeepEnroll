import numpy as np
import h5py
import re
import sys
import operator
import argparse

def load_glove_vec(fname, vocab):
    word_vecs = {}
    for line in open(fname, 'r'):
        d = line.split()
        word = d[0]

        vec = np.array(d[len(d)-300:],dtype = float)
        
        # print(list(vec))

        if word in vocab:
            word_vecs[word] = vec
    return word_vecs

def main():
  parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--dictionary', help="*.dict file", type=str,
                      default='data/entail.word.dict')
  parser.add_argument('--glove', help='pretrained word vectors', type=str, default='')
  parser.add_argument('--outputfile', help="output hdf5 file", type=str,
                      default='data/glove.hdf5')
  
  args = parser.parse_args()
  vocab = open(args.dictionary, "r").read().split("\n")[:-1]
  vocab = map(lambda x: (x.split()[0], int(x.split()[1])), vocab)
  word2idx = {x[0]: x[1] for x in vocab}
  print("word2idx size",len(word2idx))
  print("vocab size is " + str(len(word2idx)))
  w2v_vecs = np.random.normal(size = (len(word2idx), 300))
  # print("w2v_vecs.size()",w2v_vecs)
  w2v = load_glove_vec(args.glove, word2idx)
      
  print("num words in pretrained model is " + str(len(w2v)))
  for word, vec in w2v.items():
      # print("word2idx[word]",word2idx[word])
      test = word2idx[word]-1
      print(type(test))
      print("vec:",list(vec))
      print(type(vec))
      w2v_vecs[test] = vec
  for i in range(len(w2v_vecs)):
      w2v_vecs[i] = w2v_vecs[i] / np.linalg.norm(w2v_vecs[i])
  with h5py.File(args.outputfile, "w") as f:
    f["word_vecs"] = np.array(w2v_vecs)
    
if __name__ == '__main__':
    main()
