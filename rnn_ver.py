#####################################################################################
# Final Project
# Name: 杨浩伦
# ID: 2024312462
#####################################################################################
# This is the RNN translator training source codes
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch
from torch import nn
import time
import random
import numpy as np

try:
    import pickle
except ModuleNotFoundError:
    import cloudpickle as pickle


class Translation_Dataset(Dataset):
    """
    This class loads the corpus data and automatically preprocesses the raw sentences into organized training data,
    and creates an English vocabulary and Spanish vocabulary for words and punctuations occurred in the dataset.
    It's a subclass of torch.utils.data.Dataset. Later we can directly wrap it up with a DataLoader, which randomly
    separate the samples into mini batches.
    """

    def __init__(self, seq_length=10, freq_threshold=0, sample_size=None):
        """
        Initialize the dataset, and preprocess the sentences using the following steps:
        1. Remove irrelevant strings, like where the data comes from.
        2. Separate each line into English sentences and Spanish sentences, thus create two lists storing sentences.
        3. Replace non-breaking space with regular space, so nothing would go wrong.
        4. Add spaces around each punctuation, treating them as tokens.
        5. Delete consecutive spaces that may occur in the last step.
        6. create two vocabularies (dict), so we can easily convert tokens into indices, and indices into tokens.
        7. Truncate the long sentences and pad the short sentences

        :argument seq_length (int): the length of each sentence, if len > seq_length, truncate; else pad.
        :argument freq_threshold (int): the frequency threshold when creating the vocabulary.
        :argument sample_size (int): the number of samples to use when you feel 142928 pieces of data is too much.
        """
        self.seq_length = seq_length
        with open('./spa-eng/spa.txt', 'r', encoding='utf-8') as f:
            text = f.readlines()
        self.eng_text = []
        self.spa_text = []

        if sample_size is not None:
            text = text[:sample_size]

        for i in range(len(text)):
            text[i] = text[i].split('CC')[0]
            self.eng_text.append(text[i].split('\t')[0].lower())
            self.spa_text.append(text[i].split('\t')[1].lower())
        # Step 3
        self.eng_text = Translation_Dataset.remove_nbsp(self.eng_text)
        self.spa_text = Translation_Dataset.remove_nbsp(self.spa_text)
        # Step 4
        self.eng_text = Translation_Dataset.add_space_around_punctuations(self.eng_text)
        self.spa_text = Translation_Dataset.add_space_around_punctuations(self.spa_text)
        # Step 5
        self.eng_text = Translation_Dataset.delete_multiple_spaces(self.eng_text)
        self.spa_text = Translation_Dataset.delete_multiple_spaces(self.spa_text)
        # Step 6
        self.vocab_eng, self.inv_vocab_eng, self.eng_text = Translation_Dataset.create_vocab(self.eng_text,
                                                                                             freq_threshold)
        self.vocab_spa, self.inv_vocab_spa, self.spa_text = Translation_Dataset.create_vocab(self.spa_text,
                                                                                             freq_threshold)

        self.vocab_eng_size, self.vocab_spa_size = len(self.vocab_eng), len(self.vocab_spa)

        self.src_lengths = []
        self.tgt_lengths = []
        # Step 7
        for i in range(len(self.eng_text)):
            eng_tokens = self.eng_text[i].split(' ')
            spa_tokens = self.spa_text[i].split(' ')

            eng_tokens, src_length = self.truncate_or_pad(eng_tokens)
            spa_tokens, tgt_length = self.truncate_or_pad(spa_tokens)

            self.src_lengths.append(src_length)
            self.tgt_lengths.append(tgt_length)

            self.eng_text[i] = ' '.join(eng_tokens)
            self.spa_text[i] = ' '.join(spa_tokens)

    @staticmethod
    def remove_nbsp(text_list):
        """
        Removes non-breaking spaces in each sentence stored in the given list.

        :param text_list (list): A list containing the sentences.
        :return new_text_list (list): The list after removing non-breaking spaces.
        """
        new_text_list = text_list.copy()
        for i, text in enumerate(text_list):
            # \u00A0 and \u202F are codes for the non-breaking spaces
            new_text_list[i] = text_list[i].replace('\u00A0', ' ').replace('\u202F', ' ')
        return new_text_list

    @staticmethod
    def add_space_around_punctuations(text_list):
        """
        Add spaces around punctuations, so that the punctuations can also be split and recognized as tokens later.
        And add <eos> at all sentences. Notice that if a punctuation is at the head or the tail of a sentence, there is
        no need to add spaces on both ends.

        :param text_list (list): A list containing the sentences.
        :return new_text_list (list): The list after adding spaces.
        """
        all_punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~¡¿«»'
        new_text_list = text_list.copy()
        for idx, text in enumerate(text_list):
            new_text = text
            increment = 0
            for j in range(len(text)):
                if text[j] in all_punctuations:
                    if j == 0 and text[j + 1] != ' ':
                        new_text = new_text[0] + ' ' + new_text[1:]
                        increment += 1
                    elif j < len(text) - 1:
                        if new_text[j + increment - 1] != ' ':
                            new_text = new_text[:j + increment] + ' ' + new_text[j + increment:]
                            increment += 1
                        if new_text[j + increment + 1] != ' ':
                            new_text = new_text[:j + increment + 1] + ' ' + new_text[j + increment + 1:]
                            increment += 1
                    elif j == len(text) - 1 and new_text[j + increment - 1] != ' ':
                        new_text = new_text[:-1] + ' ' + new_text[-1]
            new_text = new_text + ' <eos>'
            new_text_list[idx] = new_text
        return new_text_list

    @staticmethod
    def delete_multiple_spaces(text_list):
        """
        Delete consecutive spaces that may occur in the last step.

        :param text_list (list): A list containing the sentences.
        :return new_text_list (list): The list after removing consecutive spaces.
        """
        import re
        new_text_list = text_list.copy()
        for i, text in enumerate(text_list):
            # Use regular expressions
            new_text_list[i] = re.sub(r' +', ' ', text)
        return new_text_list

    @staticmethod
    def create_vocab(text_list, freq_threshold=0):
        """
        Find the unique tokens in the whole data, count their frequencies and drop the ones with frequency <= freq_threshold.
        Then create a vocabulary of these tokens, which is in fact a dictionary. It can help us quickly convert tokens into indices
        and indices into tokens.

        :param text_list (list): A list containing the sentences.
        :param freq_threshold (int): The frequency threshold when creating the vocabulary.
        :return vocab (dict): A dictionary, its keys are tokens and its values are the indices.
        :return inv_vocab (dict): A dictionary, its keys are indices and its values are the tokens.
        :return new_text_list (list): The list containing sentences after replacing the deleted tokens with '<unk>'.
        """
        new_text_list = text_list.copy()
        token_list = [token for text in text_list for token in text.split(' ')]
        token_counter = Counter(token_list)

        vocab = {'<bos>': 0}
        inv_vocab = {0: '<bos>'}
        vocab['<pad>'] = 1
        inv_vocab[1] = '<pad>'
        index = 2
        tokens_low_freq = []
        for item, count in token_counter.items():
            if count > freq_threshold:
                vocab[item] = index
                inv_vocab[index] = item
                index += 1
            else:
                tokens_low_freq.append(item)

        for j, text in enumerate(text_list):
            tokens_of_this_line = text.split(' ')
            tokens_of_this_line_unk = [token if token not in tokens_low_freq else '<unk>' for token in
                                       tokens_of_this_line]
            new_text_list[j] = ' '.join(tokens_of_this_line_unk)
        vocab['<unk>'] = index
        inv_vocab[index] = '<unk>'

        return vocab, inv_vocab, new_text_list

    def truncate_or_pad(self, tokens):
        """
        Given a list of tokens, truncate to self.seq_length if too long, and pad to self.seq_length if too short.

        :param tokens (list): A list of tokens.
        :return: tokens (list): A list of truncated or padded tokens.
        :return: length (int): The actual, effective length of the tokens, which is of great importance later in Transformer.
        """
        if len(tokens) <= self.seq_length:
            length = len(tokens)
            tokens = tokens + ['<pad>'] * (self.seq_length - length)
        else:
            length = self.seq_length
            tokens = tokens[: self.seq_length - 1] + ['<eos>']
        return tokens, length

    def __len__(self):
        """
        Overwrite the __len__ method to return the length of the dataset, which is useful when we use DataLoader to
        generate mini-batches.

        :return: (int) The total number of samples (sentences).
        """
        return len(self.eng_text)

    def __getitem__(self, index):
        """
        Overwrite the __getitem__ method to return the indexed sample (sentence) in the dataset. The returned values are
        already converted into torch.tensor, along with their actual lengths, both English and Spanish. This will help us
        to generate mini-batches later in training.

        :param index: (int)
        :return: encoder_input (tensor): The English sentences converted into torch.tensor.
        :return: decoder_input (tensor): The Shifted Spanish sentences converted into torch.tensor, beginning with token '<bos>'
        :return: decoder_output (tensor): The original Spanish sentences converted into torch.tensor.
        :return: src_length (int): The actual length of English sentences.
        :return: tgt_length (int): The actual length of Spanish sentences.
        """
        line_eng, line_spa = self.eng_text[index], self.spa_text[index]
        tokens_eng, tokens_spa = line_eng.split(' '), line_spa.split(' ')
        src_length = self.src_lengths[index]
        tgt_length = self.tgt_lengths[index]
        tokens_spa_shifted = ['<bos>'] + tokens_spa[: -1]

        encoder_input = [self.vocab_eng[token] for token in tokens_eng]
        decoder_input = [self.vocab_spa[token] for token in tokens_spa_shifted]
        decoder_output = [self.vocab_spa[token] for token in tokens_spa]

        return torch.tensor(encoder_input, dtype=torch.long), torch.tensor(decoder_input,
                                                                           dtype=torch.long), torch.tensor(
            decoder_output, dtype=torch.long), src_length, tgt_length


class Encoder(nn.Module):
    """
    The encoder module of the Recurrent Neural Network. It uses Gated Recurrent Units (GRU) to replace the tradition rnn.
    It encodes the input (English) sentence into a hidden state, which will be the initial hidden state in the decoder.
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0., device=torch.device('cuda:0')):
        """
        Initialize the Encoder module, including the embedding layer and GRU layer.

        :param vocab_size (int): The length of the English vocabulary dict.
        :param embedding_size (int): Embedding dimension used to convert a one-hot vector to a continuous vector.
        :param hidden_size (int): The size of the hidden state.
        :param num_layers (int): The number of recurrent layers in GRU.
        :param dropout (float): The dropout probability.
        :param device: The device to use.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.rnn_layer = nn.GRU(embedding_size, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)

    def forward(self, X, hidden=None):
        """
        Forward pass of the Encoder module.

        :param X (tensor): size (batch_size, seq_length) Input
        :param hidden (tensor): size (num_layers, batch_size, hidden_size) initial hidden state
        :return: hidden (tensor): size (num_layers, batch_size, hidden_size) output hidden state of the last recurrent unit
        """
        if hidden is None:
            hidden = torch.zeros(self.num_layers, X.shape[0], self.hidden_size, device=self.device)
        X = self.embedding_layer(X)
        _, hidden = self.rnn_layer(X, hidden)  # hidden: (num_layers, batch_size, hidden_size)
        return hidden


class Decoder(nn.Module):
    """
    Decoder module of the Recurrent Neural Network. It takes in the encoded information from the encoder and uses teacher
    forcing technique to train the network to predict the next token.
    For instance:
    hidden_state + '<bos>' --> 'he'
    hidden_state + 'he' --> 'likes'
    hidden_state + 'likes' --> 'apple'
    hidden_state + 'apple' --> '.'
    hidden_state + '.' --> '<eos>'
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout=0.):
        """
        Initialize the Decoder module, including the embedding layer and GRU layer, similar to Encoder

        :param vocab_size (int): The length of the Spanish vocabulary dict.
        :param embedding_size (int): Embedding dimension used to convert a one-hot vector to a continuous vector.
        :param hidden_size (int): The size of the hidden state.
        :param num_layers (int): The number of recurrent layers in GRU.
        :param dropout (float): The dropout probability.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        # The input dimension is embedding_size + hidden_size because we will concatenate the encoded information with the decoder input as the new input.
        self.rnn_layer = nn.GRU(embedding_size + hidden_size, hidden_size, batch_first=True, num_layers=num_layers,
                                dropout=dropout)
        self.linear_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, X, hidden):
        """
        Forward pass of the Decoder module.

        :param X (tensor): size (batch_size, seq_length) Input.
        :param hidden (tensor): size (num_layers, batch_size, hidden_size) initial hidden state from the encoder.
        :return: X (tensor): size (batch_size, seq_length, vocab_size) decoder output, the logits of the output token.
        """
        context = hidden[-1]  # context: (batch_size, hidden_size)
        X = self.embedding_layer(X)  # X: (batch_size, seq_length, embedding_size)
        context = context.repeat(X.shape[1], 1, 1).transpose(0, 1)  # context: (batch_size, seq_length, hidden_size)
        X_and_context = torch.cat([X, context],
                                  dim=2)  # X_and_context: (batch_size, seq_length, embedding_size + hidden_size)
        X, _ = self.rnn_layer(X_and_context, hidden)
        X = self.linear_layer(X)
        return X


class TranslationNN(nn.Module):
    """
    Integrate Encoder and Decoder module. Build a complete Translation Recurrent Neural Network.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_size, hidden_size, num_layers=1, dropout=0.,
                 device=torch.device('cuda:0')):
        """
        Initialize the Translation RNN module. Instantiate the Encoder and Decoder modules respectively.

        :param src_vocab_size (int): the length of the English vocabulary dict.
        :param tgt_vocab_size (int): the length of the Spanish vocabulary dict.
        :param embedding_size (int): Embedding dimension used to convert a one-hot vector to a continuous vector.
        :param hidden_size (int): The size of the hidden state.
        :param num_layers (int): The number of recurrent layers in GRU.
        :param dropout (float): The dropout probability.:
        :param device: The device to use.
        """
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embedding_size, hidden_size, num_layers=num_layers, dropout=dropout,
                               device=device)
        self.decoder = Decoder(tgt_vocab_size, embedding_size, hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, encoder_input, decoder_input, hidden=None):
        """
        Forward pass of the Translation RNN module, containing both encoder and decoder.

        :param encoder_input (tensor): size (batch_size, seq_length) Input.
        :param decoder_input (tensor): size (batch_size, seq_length) Input.
        :param hidden (tensor): size (num_layers, batch_size, hidden_size) initial hidden state, usually None.
        :return: decode_output (tensor): size (batch_size, seq_length, spa_vocab_size) decoder output used to determine next tokens.
        """
        hidden = self.encoder(encoder_input, hidden)
        decoder_output = self.decoder(decoder_input, hidden)
        return decoder_output


def translator(encoder_input, dataset, net, temperature=0.5, device=torch.device('cuda:0')):
    """
    Use the trained RNN network to translate English sentence to Spanish sentence.

    :param encoder_input (tensor): size (1, seq_length) Input sentence indices.
    :param dataset (Translation_Dataset): Translation dataset instance.
    :param net (TranslationNN): TranslationNN instance.
    :param temperature (float): Temperature parameter used to adjust randomness.
    :param device: The device to use.
    :return translation (str): Spanish translation of the input sentence.
    """
    net.eval()
    encoder_input = encoder_input.to(device)
    hidden = net.encoder(encoder_input, hidden=None)
    # It needs to start with '<bos>'
    decoder_input = torch.tensor([dataset.vocab_spa['<bos>']], dtype=torch.long, device=device).reshape(1, 1)  # decoder_input: (1, 1)
    translation = ['<bos>']
    count = 0
    # Keep predicting until the ending token '<eos>' occurs.
    while dataset.inv_vocab_spa[decoder_input[0, -1].item()] != '<eos>':
        decoder_output = net.decoder(decoder_input, hidden)[0, -1, :] / temperature  # decoder_output: (1, seq_length, tgt_vocab_size)
        probabilities = torch.softmax(decoder_output, dim=0)
        next_token_idx = torch.multinomial(probabilities, 1).item()
        next_token = dataset.inv_vocab_spa[next_token_idx]
        translation.append(next_token)

        # Token by token prediction
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([next_token_idx], dtype=torch.long, device=device).reshape(1, 1)], dim=1)
        count += 1
        # Prevent the infinite loop
        if count >= dataset.seq_length * 1:
            break
    return ' '.join(translation)


def load_dataset_pickle(filepath):
    """
    Deserialize the dataset pickle file. Somtimes generate a new dataset can be time-consuming. Saving and Reading can
    save some time.

    :param filepath (str): path to the pickle file.
    :return dataset (Translation_Dataset): Translation dataset instance.
    """
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Corpus Dataset has been loaded from {filepath}")
    return dataset


def save_dataset_pickle(dataset, filepath):
    """
    Serialize the dataset into a pickle file. Sometimes generate a new dataset can be time-consuming. Saving and Reading can
    save some time.

    :param filepath (str): path to the pickle file.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Corpus Dataset has been saved to: {filepath}")


def print_progress(epoch, batch_num, total_batches, loss, bar_length=30):
    """
    A simple visualization function that prints a progress bar in each epoch.

    :param epoch (int): Current epoch index.
    :param batch_num (int):  Current batch index.
    :param total_batches (int): Total number of batches.
    :param loss (float): Current loss.
    :param bar_length (int): Length of the progress bar.
    """
    progress = (batch_num / total_batches)
    arrow = '>' * int(round(progress * bar_length) - 1)
    spaces = ' ' * (bar_length - len(arrow))

    print(f'\rEpoch {epoch}: [{arrow + spaces}] {batch_num}/{total_batches} '
          f'({progress * 100:.1f}%) Loss: {loss:.4f}', end='', flush=True)


if __name__ == '__main__':
    t1 = time.time()
    seq_length = 30
    freq_threshold = 0
    sample_size = None

    print('-' * 100)
    try:
        dataset = load_dataset_pickle('./data/dataset.pkl')
        print('Dataset already exists, loading...')
    except FileNotFoundError:
        dataset = Translation_Dataset(seq_length=seq_length, freq_threshold=freq_threshold, sample_size=sample_size)
        print(f"File not found. Creating a new dataset...")
        save_dataset_pickle(dataset, './data/dataset.pkl')
    finally:
        print(f'Loading the corpus takes {time.time() - t1:.2f} seconds')
        print(
            f'This dataset consists of {len(dataset.eng_text)} English-Spanish sentence pairs, each has length {seq_length}')
        print(
            f'After excluding the uncommon words or punctuations, the vocabulary size (including the punctuations) is {dataset.vocab_eng_size}' +
            f' for English, and {dataset.vocab_spa_size} for Spanish')

    device = torch.device("cuda:0")
    epochs = 30
    translation_net = TranslationNN(dataset.vocab_eng_size, dataset.vocab_spa_size,
                                    embedding_size=2048, hidden_size=2048,
                                    num_layers=2, dropout=0.3)
    print('-' * 100)
    try:
        translation_net.load_state_dict(torch.load('./data/translator_weights.pth'))
        print("Translation RNN weights detected, loading...")
        loss_list = np.load('./data/loss_rnn.npy').tolist()
    except (RuntimeError, FileNotFoundError) as e:
        print("Transformer weights not found. Creating a new transformer weights...")
        loss_list = []

    translation_net = translation_net.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=1)  # Ignore the padding tokens when computing loss
    optimizer = torch.optim.Adam(translation_net.parameters(), lr=1e-4)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    print('-' * 100)
    time1 = time.time()
    for epoch in range(epochs):
        if epoch % 10 == 0:
            torch.save(translation_net.state_dict(), './data/translator_weights.pth')
        translation_net.train()
        running_loss = 0.0
        batch_num = 0
        total_size = 0
        t_epoch = time.time()
        print(f"############### Epoch {epoch + 1} ###############")
        for encoder_in, decoder_in, decoder_out, _, _ in dataloader:
            encoder_in, decoder_in, decoder_out = encoder_in.to(device), decoder_in.to(device), decoder_out.to(device)
            optimizer.zero_grad()

            hidden = None
            decoder_out_hat = translation_net(encoder_in, decoder_in, hidden)
            loss = loss_fn(decoder_out_hat.reshape(-1, decoder_out_hat.shape[-1]), decoder_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(translation_net.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * encoder_in.shape[0]
            batch_num += 1
            total_size += encoder_in.shape[0]

            if batch_num % 10 == 0 or batch_num == len(dataloader):
                current_loss = running_loss / total_size
                print_progress(epoch + 1, batch_num, len(dataloader), current_loss)

        # After each epoch, make a test translation of randomly chosen sample to see the effect.
        with torch.no_grad():
            idx = random.randint(0, len(dataset.eng_text) - 1)
            original_text = ' '.join(dataset.eng_text[idx].replace('<pad>', '').split())
            actual_text = ' '.join(dataset.spa_text[idx].replace('<pad>', '').split())
            generated_text = translator(dataset[idx][0].reshape(1, -1), dataset, translation_net)
            print()
            print(f"Original Sentence: {original_text}")
            print(f"Actual Translation: {actual_text}")
            print(f"Test Translation: {generated_text}")

        print(f'One epoch took {time.time() - t_epoch:.2f} seconds')
        loss_list.append(running_loss / total_size)
        print(f"Loss: {loss_list[-1]: .6f}")

    time2 = time.time()

    print(f"Training time for {epochs} epochs: {time2 - time1: .2f}s")

    loss_hist = np.array(loss_list)
    np.save('./data/loss_rnn.npy', loss_hist)

    translation_net = translation_net.cpu()
    torch.save(translation_net.state_dict(), './data/translator_weights.pth')
    print('Net weights and loss are successfully saved to ./data/translator_weights.pth and ./data/loss_rnn.npy')

    print('Net structure:')
    print(translation_net)

