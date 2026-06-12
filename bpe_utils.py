from collections import defaultdict

def get_vocab(texts):
    vocab = defaultdict(int)
    for t in texts:
        for w in t.strip().lower().split():
            vocab[' '.join(list(w)) + ' </w>'] += 1
    return vocab

def get_stats(vocab):
    pairs = defaultdict(int)
    for w,f in vocab.items():
        s = w.split()
        for i in range(len(s)-1):
            pairs[(s[i],s[i+1])] += f
    return pairs

def merge_vocab(pair, vocab):
    bg = ' '.join(pair); rep = ''.join(pair)
    return {w.replace(bg,rep):f for w,f in vocab.items()}

def learn_bpe(texts, nm=100):
    v = get_vocab(texts); merges = []
    for _ in range(nm):
        p = get_stats(v)
        if not p: break
        b = max(p, key=p.get); merges.append(b); v = merge_vocab(b, v)
    return merges

class BPE:
    def __init__(self, merges=None):
        self.merges = merges or []
    @classmethod
    def from_texts(cls, texts, nm=100):
        return cls(learn_bpe(texts, nm))
    def encode(self, text):
        r = []
        for w in text.strip().lower().split():
            s = list(w) + ['</w>']
            for a,b in self.merges:
                i = 0
                while i < len(s)-1:
                    if s[i]==a and s[i+1]==b:
                        s = s[:i] + [a+b] + s[i+2:]
                    else: i += 1
            r.extend(s)
        return r

def build_vocab(encodings, special_tokens=None):
    if special_tokens is None:
        special_tokens = ['<pad>','<bos>','<eos>','<unk>']
    ts = set()
    for t in encodings: ts.update(t)
    vocab = {t:i for i,t in enumerate(special_tokens)}
    idx = len(special_tokens)
    for t in sorted(ts):
        if t not in vocab: vocab[t]=idx; idx+=1
    return vocab, {i:t for t,i in vocab.items()}
