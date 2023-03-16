import torch

class ChatBot(object):
    def __init__(self, searcher, idx2word, word2idx, device):
        self.device = device
        self.searcher = searcher
        self.idx2word = idx2word
        self.word2idx = word2idx
    
    def indexesFromSentence(self, sentence):
        return [self.word2idx[word] for word in sentence.split(' ')] + [2] #[EOS_token]

    def evaluate(self, sentence, max_length=15):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [self.indexesFromSentence(sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match searchers' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(self.device)
        lengths = lengths.to("cpu")
        # Decode sentence with searcher
        tokens = self.searcher(input_batch, lengths, max_length)
        # indexes -> words
        decoded_words = [self.idx2word[token.item()] for token in tokens]
        return decoded_words
    
    # Normalize input sentence and call evaluate()
    def respone(self, sentence, max_length):
        res = ' '
        try:
            output_words = self.evaluate(sentence, max_length)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            res = res.join(output_words)

        except KeyError:
            res = "Sorry i haven't learned yet"
        
        return res


        