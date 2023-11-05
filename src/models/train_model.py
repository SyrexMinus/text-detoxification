import math
import re
import time

import nltk
import numpy as np
import pandas as pd
import torch
import unicodedata
from torch import nn
from torch import optim
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from torchtext.data.metrics import bleu_score
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import tqdm

nltk.download('punkt')


TRAIN_DATA_PATH = "./data/internal/train.tsv"
VAL_DATA_PATH = "./data/internal/validatation.tsv"
BATCH_SIZE = 32
TRANSF_TRAIN_MAX_LENGTH = 15
LANG1 = "reference"
LANG2 = "translation"
REVERSE = False
DEFAULT_WORD = "this"
HIDDEN_SIZE = 128
ATTN_ENCODER_CKPT_PATH = "./models/attn_encoder_best.pt"
ATTN_DECODER_CKPT_PATH = "./models/attn_decoder_best.pt"
EPOCHS_NUM = 5
LEARNING_RATE = 0.001
SAMPLES_NUM = 3
SILENT = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(train_data_path, val_data_path):
    """Loads preprocessed data from the disk splitted into train and valibation sets"""
    train = pd.read_csv(train_data_path, sep="\t", index_col=0)
    val = pd.read_csv(val_data_path, sep="\t", index_col=0)
    return train, val


class Lang:
    SOS_token = 0
    EOS_token = 1

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.SOS_token: "SOS", self.EOS_token: "EOS"}
        self.n_words = len(self.index2word)  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in word_tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, lang2, reverse, data):
    # Split every line into pairs and normalize
    pairs = [[normalizeString(l[lang1]), normalizeString(l[lang2])] for i, l in data.iterrows()]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p, max_len):
    return len(word_tokenize(p[0])) < max_len and \
        len(word_tokenize(p[1])) < max_len


def filterPairs(pairs, max_len):
    return [pair for pair in pairs if filterPair(pair, max_len)]


def prepareData(lang1, lang2, reverse, data, max_len):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse, data)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_len)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence, default_word):
    return [lang.word2index.get(word) or lang.word2index[default_word] for word in word_tokenize(sentence)]


def tensorFromSentence(lang, sentence, device, default_word):
    indexes = indexesFromSentence(lang, sentence, default_word)
    indexes.append(Lang.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(pair, input_lang, output_lang, device, default_word):
    input_tensor = tensorFromSentence(input_lang, pair[0], device, default_word)
    target_tensor = tensorFromSentence(output_lang, pair[1], device, default_word)
    return input_tensor, target_tensor


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor, max_len, device):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(Lang.SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion, max_len):
    encoder.train()
    decoder.train()
    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor, max_len, device)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(encoder, decoder, sentence, input_lang, output_lang, device, default_word, target_tensor, max_len):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device, default_word)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden, target_tensor, max_len,
                                                                device)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == Lang.EOS_token:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


class DetoxModel:
    def __init__(self, *args, **kwargs):
        pass

    def detox(self, sentence: list[str]) -> list[str]:
        raise NotImplementedError


class TransformerDetoxModel(DetoxModel):
    def __init__(self, encoder, decoder, input_lang, output_lang, device, default_word, max_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.device = device
        self.default_word = default_word
        self.max_len = max_len

    def detox(self, sentence: list[str]) -> list[str]:
        sentence = normalizeString(" ".join(sentence))
        self.encoder.eval()
        self.decoder.eval()
        output_words, _ = evaluate(self.encoder, self.decoder, sentence, self.input_lang, self.output_lang, self.device,
                                   self.default_word, None, self.max_len)
        return output_words[:-1]


def assess_model(model: DetoxModel, samples: int, silent, val, references_corpus, lang1, lang2) -> float:
    if not silent:
        for index, row in val[:samples].iterrows():
            ref = word_tokenize(row[lang1])
            tran = word_tokenize(row[lang2])
            print(f"Reference: \"{ref}\"")
            print(f"Detoxed: \"{model.detox(ref)}\"")
            print(f"Translation: \"{tran}\"\n")
    candidate_corpus = [model.detox(word_tokenize(seq)) for seq in tqdm(val[lang1])]
    bleu = bleu_score(candidate_corpus, references_corpus)
    if not silent:
        print(f"BLEU: {bleu}")
    return bleu


def val_one_epoch(encoder, decoder, epoch_num, best_so_far, encoder_ckpt_path, decoder_ckpt_path, input_lang,
                  output_lang, device, default_word, samples, silent, references_corpus, lang1, lang2, max_len):
    print(f"Epoch {epoch_num}: val")
    metric = assess_model(TransformerDetoxModel(encoder, decoder, input_lang, output_lang, device, default_word,
                                                max_len),
                          samples, silent, val, references_corpus, lang1, lang2)
    if metric > best_so_far:
        torch.save(encoder.state_dict(), encoder_ckpt_path)
        torch.save(decoder.state_dict(), decoder_ckpt_path)
        best_so_far = metric

    return metric, best_so_far


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train_fun(train_dataloader, encoder, decoder, n_epochs, learning_rate, encoder_ckpt_path, decoder_ckpt_path,
              default_word, samples, silent, references_corpus, lang1, lang2, max_len):
    start = time.time()
    best_so_far = -1.0
    losses = []
    metrics = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len)
        print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                    epoch, epoch / n_epochs * 100, loss))
        metric, best_so_far = val_one_epoch(encoder, decoder, n_epochs, best_so_far, encoder_ckpt_path,
                                            decoder_ckpt_path, input_lang, output_lang, device, default_word, samples,
                                            silent, references_corpus, lang1, lang2, max_len)
        metrics.append(metric)
        losses.append(loss)

    return losses, metrics


def get_dataloader(batch_size, max_len, device, lang1, lang2, reverse, data, default_word):
    input_lang, output_lang, pairs = prepareData(lang1, lang2, reverse, data, max_len)

    n = len(pairs)
    input_ids = np.zeros((n, max_len), dtype=np.int32)
    target_ids = np.zeros((n, max_len), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp, default_word)
        tgt_ids = indexesFromSentence(output_lang, tgt, default_word)
        inp_ids.append(Lang.EOS_token)
        tgt_ids.append(Lang.EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader


if __name__ == "__main__":
    print("Load data...")
    train, val = load_data(TRAIN_DATA_PATH, VAL_DATA_PATH)
    print("Preprocess data...")
    input_lang, output_lang, train_dataloader = get_dataloader(
        BATCH_SIZE, TRANSF_TRAIN_MAX_LENGTH, device, LANG1, LANG2, REVERSE, train, DEFAULT_WORD
    )
    print("Init encoder...")
    encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
    print("Init decoder...")
    decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words).to(device)
    print("Extract references_corpus...")
    references_corpus = [[word_tokenize(seq)] for seq in val[LANG2]]
    print("Train...")
    train_fun(train_dataloader, encoder, decoder, EPOCHS_NUM, LEARNING_RATE, ATTN_ENCODER_CKPT_PATH,
              ATTN_DECODER_CKPT_PATH, DEFAULT_WORD, SAMPLES_NUM, SILENT, references_corpus, LANG1, LANG2,
              TRANSF_TRAIN_MAX_LENGTH)
    print("Finish")