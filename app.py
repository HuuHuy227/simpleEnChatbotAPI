from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import torch
import torch.nn as nn
from model.Seq2Seq import Searcher
from model.Seq2Seq import EncoderRNN
from model.Seq2Seq import AttnDecoderRNN
from utils import ChatBot
import pickle

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_weigth = './weights/encoder.pt'
decoder_weigth = './weights/decoder.pt'
idx2word_path = './weights/idx2word.pkl'
word2idx_path = './weights/word2idx.pkl'

with open(idx2word_path, 'rb') as handle:
    idx2word = pickle.load(handle)

with open(word2idx_path, 'rb') as handle:
    word2idx = pickle.load(handle)

output_size = len(idx2word)
hidden_size = 256
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1

def load_model():
    encoder = EncoderRNN(output_size, hidden_size, encoder_n_layers, dropout)
    decoder = AttnDecoderRNN(hidden_size, output_size, decoder_n_layers, dropout)

    encoder.load_state_dict(torch.load(encoder_weigth,map_location = DEVICE)) 
    decoder.load_state_dict(torch.load(decoder_weigth,map_location = DEVICE)) 

    searcher = Searcher(encoder, decoder , DEVICE)
    chatbot = ChatBot(searcher, idx2word, word2idx, DEVICE)

    return chatbot

app = FastAPI(debug = True)
origins = ["*"]

app.add_middleware(  # Enable CORS
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

chatbot = load_model()

class body_request(BaseModel):
    sentence: str
    max_token: int

@app.get("/")
async def home():
    return {"text" : "Hello word"}

@app.post("/respone")
async def predict(item: body_request):
    # Make prediction
    respone = chatbot.respone(item.sentence, item.max_token)
    return {"code" : 0, "result" : respone}

if __name__ == '__main__':
    uvicorn.run(app)