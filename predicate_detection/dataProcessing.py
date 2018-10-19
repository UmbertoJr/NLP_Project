import numpy as np

class read_file:
    def __init__(self, name_file):
        self.name = name_file
        self.position_file_byte = 0
    
    def read_sentences(self, num_sent):
        with open(self.name, "r") as f:
            f.seek(self.position_file_byte)
            docs = []
            sentence = []
            m = 0
            line = f.readline()
            while line and len(docs) < num_sent:
                word = line.split("\t")
                if word[0] != "\n":
                      if  int(word[0]) > m:
                            sentence.append(word)
                            line = f.readline()
                            m += 1

                else:
                    m = 0
                    docs.append(sentence)
                    sentence = []
                    line = f.readline()
                    if line == "":
                        t.seek(0)
            self.position_file_byte = f.tell()
    
        return docs
    
    def print_sentence(self, sent):
        print(*[sent[i][1]for i in range(len(sent))], sep=" ")
        
        
    def find_all_predicates(self, occ_greater_than):
        dic = {}
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    continue
                predicate = line.split()[13]
                if predicate in dic:
                    dic[predicate] +=1
                else:
                    dic[predicate] = 1

        dic_final_to_fit = {el for el in dic if dic[el]> occ_greater_than}
        dic_final_not_to_fit = {el for el in dic if dic[el] <= occ_greater_than}
        return dic_final_to_fit, dic_final_not_to_fit
    
    
    
    def find_all_POS(self):
        dic = {}
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    continue
                pos = line.split()[4]
                if pos in dic:
                    dic[pos] += int(bool(line.split()[13]!= "_"))
                else:
                    dic[pos] = int(bool(line.split()[13]!= "_"))
        pos_pos = {}
        j = 0
        for i in dic.keys():
            pos_pos[i] = j
            j+= 1
        return dic, pos_pos

    
    def max_length_sentence(self):
        m = 0
        j = 0
        with open(self.name, "r") as f:
            while True:
                line = f.readline()
                j += 1
                if line =="":
                    break   
                if  line.split("\t")[0]=="\n":
                    if j >m :
                        m = j
                    j = 0
                    continue
        return m
    
    
    
class model_builder:
    # costruisce il DATASET con le parole sotto forma di vettore e definisce il modello con gli embeddings
    
    def __init__(self, pos_pos, max_length):
        self.model = {}
        self.name_file = ""
        self.max_length = max_length
        self.len_pos = len(pos_pos) ## numero di possibili POS   
        self.pos_pos = pos_pos
        
        
    def load_model(self, name_file):
        ### salva gli embeddings dal file
        ## ossia crea un dizionario con key il lemma e valore il byte nel file (in maniera tale da accedere al file più rapidamente)
        self.name_file = name_file
        with open(name_file, "r") as f:
            pos = f.tell()
            while True:
                line = f.readline()
                if line == "":
                    break
                word = line.split()[0]
                self.model[word] = pos
                pos = f.tell()
                
        return self.model
    
    def __call__(self, word):
        ### tramite una chiamata ritorna l'embeddings della parola cercata
        with open(self.name_file, "r") as f:
            f.seek(self.model[word])
            return np.array(f.readline().split()[1:])
        
   
    def create_we_for(self, sent):
        #ritorna per ciascuna sentence un array dim (len_sentence + pad, dim_embeddings) 
        # [len_sentence + pad = max_lenght(valore sccelto 50)]
        #   e un intero con valore la lunghezza della sentence "n"
        n = len(sent)
        w=0
        Sent = []
        while w < len(sent) :
            lemma = sent[w][2]
            try:
                Sent.append(self(lemma))
            except:
                Sent.append(self("unk"))
            w+=1           
            
        sent_x = np.pad(np.array(Sent, dtype = np.float32),((0,self.max_length-n),(0,0)), "constant", constant_values = 0 )
        return (sent_x ,  n)
                
    def creation(self, sents, max_length):
        ## ritorna per un batch di sentence un array dim [batch, max_length, dim_embeddings]
        # e un array dim [batch] con la lunghezza di ciascuna sentence
        self.max_length = max_length
        X = []
        seq_len = []
        P = []
        POS = []
        for sent in sents:
            if len(sent) > max_length:
                r = (len(sent)-50)%20
                if r != 0:
                    sent += [['_' for _ in range(len(sent[0]))] for _ in range(20-r)]                          #####   DEVO CONTROLLARE
                    sents_after_conv_win = [sent[20*i: 20*i+50] for i in range(len(sent)//20)]                 ##### CHE FUNZIONI BENE
                for s in sents_after_conv_win:
                    x, n = self.create_we_for(s)
                    p = self.is_pred(s)
                    pos = self.pos_arg(s)
                    X.append(x), seq_len.append(n), P.append(p), POS.append(pos)

            else:
                x, n = self.create_we_for(sent)
                p = self.is_pred(sent)
                pos = self.pos_arg(sent)
                X.append(x), seq_len.append(n), P.append(p), POS.append(pos)
        return np.array(X), np.array(seq_len), np.array(P), np.array(POS)
    
    
    def is_pred(self, sent):
        n = len(sent)
        bool_pred = [0 for _ in range(self.max_length)]
        for w in range(n):
            if sent[w][13]!= '_':
                bool_pred[w] = 1
        one_hot_pred = np.zeros((self.max_length, 2 ))
        one_hot_pred[np.arange(self.max_length), bool_pred] = 1
        return one_hot_pred
    
    def pos_arg(self, sent):
        vec_pos = []
        p = 0
        for w in sent:
            if w[4]!='_':
                vec_pos.append(self.pos_pos[w[4]])
            else:
                p += 1
        one_hot_pos = np.zeros((self.max_length, self.len_pos ))
        one_hot_pos[np.arange(len(sent)-p), vec_pos] = 1
        return one_hot_pos
            
            