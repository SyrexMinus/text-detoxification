import argparse

import torch
from nltk import word_tokenize

from train_model import (load_data, TRAIN_DATA_PATH, VAL_DATA_PATH, get_dataloader, BATCH_SIZE,
                         TRANSF_TRAIN_MAX_LENGTH, device, LANG1, LANG2, REVERSE, DEFAULT_WORD, EncoderRNN,
                         HIDDEN_SIZE, AttnDecoderRNN, ATTN_ENCODER_CKPT_PATH, ATTN_DECODER_CKPT_PATH,
                         TransformerDetoxModel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='predict_model',
        description='Detoxifies a given sentence')
    parser.add_argument('sentence', type=str, help='a sentence to check')
    args = parser.parse_args()
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
    print("Load encoder weights...")
    if device.type == "cpu":
        ckpt = torch.load(ATTN_ENCODER_CKPT_PATH, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(ATTN_DECODER_CKPT_PATH)
    encoder.load_state_dict(ckpt)
    print("Load decoder weights...")
    if device.type == "cpu":
        ckpt = torch.load(ATTN_DECODER_CKPT_PATH, map_location=torch.device('cpu'))
    else:
        ckpt = torch.load(ATTN_DECODER_CKPT_PATH)
    decoder.load_state_dict(ckpt)
    print("Create model wrapper...")
    attn_trans_model = TransformerDetoxModel(encoder, decoder, input_lang, output_lang, device, DEFAULT_WORD,
                                             TRANSF_TRAIN_MAX_LENGTH)
    ref = word_tokenize(args.sentence)
    print(" ".join(attn_trans_model.detox(ref)))