import torch
import  transformer_ver as tfm
import rnn_ver as rnn
from rnn_ver import Translation_Dataset


if __name__ == "__main__":
    device = torch.device('cpu')
    # ---------------------------RNN Translation Network Configuration---------------------------
    embedding_size = 4000
    hidden_size = 3000
    num_layers = 2
    dropout = 0.3
    seq_len = 20
    sample_size = None
    freq_threshold = 0
    dataset_rnn = Translation_Dataset(seq_length=seq_len, sample_size=sample_size, freq_threshold=freq_threshold)
    translation_net = rnn.TranslationNN(dataset_rnn.vocab_eng_size, dataset_rnn.vocab_spa_size, embedding_size=embedding_size,
                                            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, device=device)
    try:
        translation_net.load_state_dict(torch.load('./data/translator_weights.pth'))
        print('RNN: <All keys matched successfully>')
    except (RuntimeError, FileNotFoundError):
        print('RNN translator weights not found, please check your path.')

    translation_net.to(device)
    # ---------------------------Transformer Translation Network Configuration---------------------------
    num_encoders, num_decoders = 2, 2
    hidden_size = 2048
    seq_length = 30
    sample_size = None
    freq_threshold = 0
    dropout = 0.
    dataset = tfm.Translation_Dataset(seq_length=seq_length, sample_size=sample_size, freq_threshold=freq_threshold)

    transformer = tfm.Transformer(num_encoders, num_decoders,
                              dataset.vocab_eng_size, dataset.vocab_spa_size,
                              seq_len=seq_length, hidden_size=hidden_size, dropout=dropout, device=device)
    try:
        transformer.load_state_dict(torch.load('./data/transformer_weights_cpu.pth'))
        print('Transformer: <All keys matched successfully>')
    except FileNotFoundError:
        print('Transformer translator weights not found, please check your path.')

    transformer = transformer.to(device)
    # ---------------------------Begin Translating---------------------------
    while True:
        text0 = input('Please enter the sentence you want to translate (type exit to quit): ')
        try:
            if text0 == 'exit':
                print('Goodbye...')
                break
            text = Translation_Dataset.remove_nbsp([text0.lower()])
            text = Translation_Dataset.add_space_around_punctuations(text)
            text = Translation_Dataset.delete_multiple_spaces(text)
            text = text[0]
            text = text.split(' ')
            tokens = torch.tensor([dataset_rnn.vocab_eng[token] for token in text], dtype=torch.long)
            print('    RNN     translation: ' + rnn.translator(tokens.reshape(1, -1), dataset_rnn, translation_net, temperature=0.5, device=device))
            tokens = torch.tensor([dataset.vocab_eng[token] for token in text], dtype=torch.long)
            print('Transformer translation: ' + tfm.translator(tokens.reshape(1, -1), dataset, transformer, temperature=0.5, device=device))

        except KeyError as e:
            print(f'Sorry, \"{e.args[0]}\" isn\'t in our vocabulary.')

