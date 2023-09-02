from csv import list_dialects
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
from torch import nn, from_numpy, tensor, cat, mean
from os import listdir, path
import sys
from utils.file_io import load_yaml_file
from pathlib import Path

sys.path.append('/home/qust-011/caption/dep_baseline/dep_parser/')
from pygcn_master.pygcn import GCN

nlp = StanfordCoreNLP(r'/home/qust-011/corenlp/stanford-corenlp-4.5.0')
t = 'A machine whines and squeals while rhythmically punching or stamping'.lower()
f_vocab, f_embs = 'vocab_npa_{}.npy', 'embs_npa_{}.npy'
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPE_TOKENS = [PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX]
# print('original caption:', t)

settings = load_yaml_file(Path('./exp_settings/dcb.yaml'))

def create_embs_vocab(glove_len):
    vocab,embeddings = [],[]
    glove_path = '/home/qust-011/ml_models/glove/glove.6B.{}d.txt'.format(glove_len)
    
    with open(glove_path,'rt',encoding='utf-8') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
        
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings) 
    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, PAD_IDX, '<pad>')
    vocab_npa = np.insert(vocab_npa, UNK_IDX, '<unk>')
    vocab_npa = np.insert(vocab_npa, SOS_IDX, '<sos>')
    vocab_npa = np.insert(vocab_npa, EOS_IDX, '<eos>')

    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.
    sos_emb_npa = unk_emb_npa/2
    eos_emb_npa = unk_emb_npa/3

    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, sos_emb_npa, eos_emb_npa, embs_npa))
    # vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    # save local file
    v_file = f_vocab.format(glove_len)
    e_file = f_embs.format(glove_len)
    with open(v_file,'wb') as f:
        np.save(f,vocab_npa)
    with open(e_file,'wb') as f:
        np.save(f,embs_npa)
        
    print(v_file, e_file, 'cached.')

def get_embs(glove_len):
    e_file = f_embs.format(glove_len)
    if path.exists(e_file):
        return np.load(e_file)
    
    create_embs_vocab(glove_len)
    return np.load(e_file)

def get_vocab(glove_len):
    v_file = f_vocab.format(glove_len)
    if path.exists(v_file):
        return np.load(v_file)
    
    create_embs_vocab(glove_len)
    return np.load(v_file)

def get_glove_embed_layer(glove_len):
    # try loading local file
    e_file, v_file = f_embs.format(glove_len), f_vocab.format(glove_len)
    if path.exists(e_file) and path.exists(v_file):
        return nn.Embedding.from_pretrained(from_numpy(get_embs(glove_len)).float())
    
    create_embs_vocab(glove_len)
    return nn.Embedding.from_pretrained(from_numpy(get_embs(glove_len)).float())

def pad_adjmat(ret): 
    seq_len = ret.shape[0]
    seq_max = settings['data']['max_caption_tok_len']
    
    # truncate
    if seq_len > seq_max:
        return ret[:seq_max, :seq_max]
    # padding
    elif seq_len < seq_max:
        for _ in range(seq_max - seq_len):
            # add row with zeros
            ret = np.insert(ret, ret.shape[0], np.zeros(ret.shape[1]), axis=0)
            # add column with zeros
            ret = np.insert(ret, ret.shape[1], np.zeros(ret.shape[0]), axis=1)
    return ret

# Returns dependency adj mat with an extra global node.
def text2depAdj(txt: str, use_padding=False):
    tokens = nlp.word_tokenize(txt)
    deps = nlp.dependency_parse(txt)
    
    # print(deps)
    
    # get head list
    head_list = []    
    for i in range(len(tokens)):
        for d in deps:
            if d[-1] == i + 1:
                head_list.append(int(d[1]))
                
    # head list to adj mat
    ret = np.zeros((len(tokens), len(tokens)), dtype=np.float32)
    for i in range(len(head_list)):
        j=head_list[i]
        if j!=0:
            ret[i,j-1]=1
            ret[j-1,i]=1
            
    # add global context node
    # add row with ones
    ret = np.insert(ret, 0, np.ones(ret.shape[1]), axis=0)
    # add column with ones
    ret = np.insert(ret, 0, np.ones(ret.shape[0]), axis=1)
    ret[0][0] = 0
    
    if use_padding:
        ret = pad_adjmat(ret)

    return ret, tokens
    
vocab = None
vocab_dic = None
vocab_loaded=False

def get_vocab_dic(glove_len):
    if not path.exists(f_vocab):
        create_embs_vocab(glove_len)
        
    vocab = np.load(f_vocab)
    vocab_dic = {}
    for i in range(len(vocab)):
        vocab_dic[vocab[i]] = i
    return vocab_dic

def token2ids(tokens, glove_len, config=None):
    global vocab
    global vocab_dic
    global vocab_loaded
    idx_list = []
    
    v_file = f_vocab.format(glove_len)

    if not path.exists(v_file):
        create_embs_vocab(glove_len)

    if not vocab_loaded:
        vocab = np.load(v_file)
        vocab_dic = {}
        for i in range(len(vocab)):
            vocab_dic[vocab[i]] = i
        vocab_loaded = True
        
    for to in tokens:
        try:
            idx_list.append(vocab_dic[to])
        except:
            idx_list.append(UNK_IDX)
            
    if len(idx_list) > config.cap_max_len:
        return idx_list[:config.cap_max_len]

    for _ in range(0, config.cap_max_len -len(idx_list)):
        idx_list.append(PAD_IDX)
    return idx_list

# def text2ids(text):
    

def caption2dep_embed(capt, config):
    adj, tokens = text2depAdj(t)
    em_layer = get_glove_embed_layer(config.decoder.nhid)
    idx = token2ids(tokens)  
    
    embeds = em_layer(tensor([idx]))[0]
    embeds = cat([mean(embeds, dim=0).view(1,-1), embeds])
    
    gcn = GCN(embeds.shape[1], config.gcn_out_len, 2, 0.5)
    return gcn(embeds, tensor(adj))[1][0]

if __name__ == '__main__':
    print('dependency tree:\n', nlp.parse(t))
    adj, tokens = text2depAdj(t)
    print('adj matrix:\n',adj)
    print('tokens:', tokens)

    em_layer = get_glove_embed_layer()
    idx = token2ids(tokens)
    print('token idx:', idx)
    print('glove embed layer loaded.')

    embeds = em_layer(tensor([idx]))[0]
    embeds = cat([mean(embeds, dim=0).view(1,-1), embeds])
    # print('embeds:\n', embeds)
    print('embeds shape:', embeds.shape)
    
    gcn = GCN(embeds.shape[1], 80, 2, 0.5)
    output = gcn(embeds, tensor(adj))
    print('global context embed:\n', output[1][0])