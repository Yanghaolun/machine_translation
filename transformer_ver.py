#####################################################################################
# Final Project
# Name: 杨浩伦
# ID: 2024312462
#####################################################################################
# This is the transformer translator training source code
from torch.utils.data import DataLoader
import torch
from torch import nn
import time
import random
import numpy as np
from rnn_ver import Translation_Dataset, save_dataset_pickle, load_dataset_pickle, print_progress
try:
    import pickle
except ModuleNotFoundError:
    import cloudpickle as pickle


class PositionEncoding(nn.Module):
    """
    Positional encoding module, using the method originally in Vaswani et al., 2017
    p_{i,2j} = sin(i /  10000^{2j/d})
    p_{i,2j+1} = cos(i /  10000^{2j/d})
    This information, embedded in the input tensor, can help encoder and decoder identify the correct order of the input tokens.
    """
    def __init__(self, vocab_size, seq_len=100, dropout=0., device=torch.device('cpu')):
        """
        Initialize the positional encoding module. Create the embedding tensor P, whose entries are defined as above.
        Notice that this tensor P remain constant no matter what this input is. So we can create a sufficiently large
        P. If we need, we can slice and embed P[:seq_length, :vocab_size]. This way, we don't need to recompute P every
        time we need.

        :param vocab_size (int): The size of the vocabulary dictionary.
        :param seq_len (int): The length of the input sequence.
        :param dropout (float): The dropout probability.
        :param device: The device to use.
        """
        super().__init__()
        exponent = torch.ceil(torch.tensor(list(range(0, vocab_size))) / 2) * 2
        exponent = exponent.reshape(1, -1)
        rad = torch.tensor(list(range(1, seq_len + 1))).reshape(-1, 1) / torch.pow(1e4, exponent / vocab_size)
        self.P = torch.zeros_like(rad).to(device)
        self.P[:, 0::2] = torch.cos(rad[:, 0::2])
        self.P[:, 1::2] = torch.sin(rad[:, 1::2])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        """
        Forward pass of the positional encoding module. P has dimension (seq_len, vocab_size) we need to broadcast
        it over the batch_size dimension. Unsqueeze to add an axis 0, so P will broadcast over the batch_size dimension.
        ':X.shape[1]' on axis=1 is necessary. Although during the training process, each sample's length = seq_len because
        of truncation or padding. But when making predictions, the length of the decoder input is ascending step by step.
        So we need to slice P to guarantee stable output.

        :param X (tensor): size (batch_size, seq_len, vocab_size) Input
        :return X (tensor): size (batch_size, seq_len, vocab_size) Output
        """
        return self.dropout(X) + self.P.unsqueeze(dim=0)[:, :X.shape[1], :]


class Attention(nn.Module):
    """
    The famous attention module in the transformer. it takes in the queries, keys and values and outputs the results
    of each query. We also need to consider the mask. Some key-value pairs are not considered. We use the dot product
    attention, namely, output = mask(softmax(QK^T / sqrt(d))) * V.
    """
    def __init__(self, dropout=0.):
        """
        Initialize the attention module. The attention module doesn't have any parameters that need to be updated.
        It just contains a softmax layer and dropout layer. The other operations are just tensor product.

        :param dropout (float): The dropout probability.
        """
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, valid_len=None):
        """
        The forward pass of the attention module. There three kinds of masks
        1. valid_len == None: No mask
        2. valid_len.dim() == 1: all queries within one sample (sentence) share the same mask, usually used in encoder,
           since every token within one sentence is considered known information.
        3. valid_len.dim() == 2: each query within one sample (sentence) can have different mask, usually used in decoder,
           since every token is only allowed to see the tokens before itself, in order to make token-by-token predictions.

        :param query (tensor): size (batch_size, num_queries, hidden_size) Input.
        :param key (tensor):  size (batch_size, num_keys, hidden_size) Input.
        :param value (tensor): size (batch_size, num_keys, hidden_size) Input.
        :param valid_len (tensor): size (batch_size, ) or (batch_size, num_queries) Input.
        :return Y (tensor): size (batch_size, num_queries, hidden_size) Output.
        """
        if valid_len is None:
            prob = self.softmax(torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(query.shape[-1])))

        elif valid_len.dim() == 1:
            # valid_len: (batch_size, ) - different queries in one sample share the same valid length
            attention_weight = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(query.shape[-1]))
            col_indices = torch.arange(attention_weight.shape[2])
            mask = col_indices < valid_len.unsqueeze(1)
            mask = torch.repeat_interleave(mask.unsqueeze(dim=1), query.shape[1], dim=1)
            attention_weight[~mask] = -1000
            prob = self.softmax(attention_weight)

        elif valid_len.dim() == 2:
            # valid_len: (batch_size, num_queries) - different queries in one sample have different lengths
            attention_weight = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(query.shape[-1]))
            col_indices = torch.arange(attention_weight.shape[2])
            mask = col_indices < valid_len.unsqueeze(-1)
            attention_weight[~mask] = -1000
            prob = self.softmax(attention_weight)

        Y = torch.bmm(self.dropout(prob), value)
        return Y


class MHAttention(nn.Module):
    """
    Multi-head attention module in the transformer. It takes in the queries, keys and values, use fully connected layers
    to make them have the same size, split them into multiple heads, and carry out the attention layer parallely.
    """
    def __init__(self, query_size, key_size, value_size, hidden_size, num_heads=1, dropout=0.):
        """
        Initialize the multihead attention module. Use linear layers to make them have the same size, also make the model
        more expressive.

        :param query_size (int): The size of the input query.
        :param key_size (int): The size of the input key.
        :param value_size (int): The size of the input value.
        :param hidden_size (int): The size of the hidden state. All the Q, K and Vs will have this size later.
        :param num_heads (int): The number of attention heads.:
        :param dropout (float): The dropout probability.
        """
        super().__init__()
        self.linear_Q = nn.Linear(query_size, hidden_size, bias=False)
        self.linear_K = nn.Linear(key_size, hidden_size, bias=False)
        self.linear_V = nn.Linear(value_size, hidden_size, bias=False)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention = Attention(dropout=dropout)

    def separate_multiheads(self, X):
        """
        Given queries or keys or values, separate them into multiple heads. And combine multiple heads over the batch_size
        dimension. That is to say, treat different heads as if they were another samples, so the attention calculation can be
        carried out parallely. An example of what this function does is:
        X = [[[1, 2, 3, 4],
              [5, 6, 7, 8]],
             [[9, 10, 11, 12],
              [13, 14, 15, 16]]]
        Now, X.shape = [2, 2, 4], if we separate X into 2 heads, we will get
        X_multiheads = [[[1, 2],
                         [5, 6]],
                        [[3, 4],
                         [7, 8]],
                        [[9, 10],
                         [13, 14]],
                        [[11, 12],
                         [15, 16]]]
        It has shape [4, 2, 2].

        :param X (tensor): size (batch_size, num_queries/keys/values, hidden_size) Input.
        :return: X_multiheads (tensor): size (batch_size * num_heads, num_queries/keys/values, hidden_size / num_heads) Output.
        """
        batch_size, seq_len = X.shape[0], X.shape[1]
        # Step 1: swap seq_len axis and hidden_size axis, so that when we reshape, the order is correct
        X_trans = X.transpose(1, 2)
        # Step 2: Cut samples into multiheads
        X_separate_heads = X_trans.reshape(batch_size, self.num_heads, -1, seq_len)
        # Step 3: Restore the correct order of seq_len axis and hidden_size axis
        X_separate_heads = X_separate_heads.transpose(2, 3)
        # Step 4: Combine the head axis with the batch axis, treat different heads as different samples
        return X_separate_heads.reshape(batch_size * self.num_heads, seq_len, -1)

    def combine_multiheads(self, Y):
        """
        Reverse the operation in self.separate_multiheads(). After separating the queries, keys and values, and use
        them to compute attention. We need to restore the outputs into correct shape.

        :param Y (tensor): size (batch_size * num_heads, num_queries/keys/values, hidden_size / num_heads) Input.
        :return Y_combined (tensor): size (batch_size, num_queries/keys/values, hidden_size) Output.
        """
        # Step 1: Separate batch axis and head axis
        batch_size, seq_len = int(Y.shape[0] / self.num_heads), Y.shape[1]
        Y_multiheads = Y.reshape(batch_size, self.num_heads, seq_len, -1)
        # Step 2: Swap seq_len axis and hidden_size axis, so that when we combine different heads, the order is correct
        Y_trans = Y_multiheads.transpose(2, 3)
        # Step 3: Combine different heads
        Y_combined = Y_trans.reshape(batch_size, -1, seq_len)
        # Step 4: Restore seq_len axis and hidden_size axis
        return Y_combined.transpose(1, 2)

    def forward(self, query, key, value, valid_len=None):
        """
        Forward pass of the multihead attention module.

        :param query (tensor): size (batch_size, num_queries, query_size) Input.
        :param key (tensor): size (batch_size, num_keys, key_size) Input.
        :param value (tensor): size (batch_size, num_keys, value_size) Input.
        :param valid_len (tensor): size (batch_size, ) or (batch_size, num_queries) Input.
        :return Y (tensor): size (batch_size, num_queries, hidden_size) Output.
        """
        query = self.linear_Q(query)
        key = self.linear_K(key)
        value = self.linear_V(value)
        # Now, query/key/value size: (batch_size, num_queries/keys/values, hidden_size)
        query_multiheads = self.separate_multiheads(query)
        key_multiheads = self.separate_multiheads(key)
        value_multiheads = self.separate_multiheads(value)
        # Note that the valid lengths also need to change. One sample now becomes num_heads samples. So the valid lengths
        # also need to repeat num_heads times.
        if valid_len is not None:
            valid_len = torch.repeat_interleave(valid_len, self.num_heads, dim=0)
        Y = self.attention(query_multiheads, key_multiheads, value_multiheads, valid_len)
        return self.combine_multiheads(Y)


class Add_and_Norm(nn.Module):
    """
    Residual addition and layer normalization After each attention or FFN layer. A standard practice in Transformer.
    """
    def __init__(self, hidden_size, dropout=0.):
        """
        Initialize the layer normalization.

        :param hidden_size (int): The size of each token vector.
        :param dropout (float): The dropout probability.
        """
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Y, X):
        """
        Forward pass of the layer. First add residual, then do layer normalization.
        :param Y (tensor): size (batch_size, num_queries, hidden_size) The output of attention or FFN layer.
        :param X (tensor): size (batch_size, num_queries, hidden_size) The input of attention or FFN layer.
        :return Y (tensor): size (batch_size, num_queries, hidden_size) Output.
        """
        return self.layernorm(self.dropout(Y) + X)


class FFN(nn.Module):
    """
    Position-wise Feed-Forward Neural Network, which is in fact a fully-connected layer used on each query.
    All queries share the same weight and bias. In order to make the network more expressive, I use 2 linear layers,
    with ReLU() as activation.
    """
    def __init__(self, hidden_size, dropout=0.):
        """
        Initialize the Feed-Forward Neural Network, including two fully-connected layers followed by a ReLU activation.

        :param hidden_size (int): The size of each query(token) vector.
        :param dropout (float): The dropout probability.
        """
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        """
        The forward pass of the FFN layer.

        :param X (tensor): size (batch_size, num_queries, hidden_size) Input.
        :return Y (tensor): size (batch_size, num_queries, hidden_size) Output.
        """
        X = self.linear1(X)
        X = self.relu(X)
        X = self.linear2(X)
        return self.relu(self.dropout(X))


class EncoderBlock(nn.Module):
    """
    Build the encoder block in the Transformer. It includes 2 sub-layers:
    1. Self-Multihead-attention --> Add & LayerNorm
    2. Feed-forward --> Add & LayerNorm
    """
    def __init__(self, query_size, key_size, value_size, hidden_size, num_heads=1, dropout=0.):
        """
        Initialize the encoder block.

        :param query_size (int): The size of the query vector.
        :param key_size (int): The size of the key vector.
        :param value_size (int): The size of the value vector.
        :param hidden_size (int): The size of the vector which we are going to convert queries/keys/values into.
        :param num_heads (int): The number of attention heads.
        :param dropout (float): The dropout probability.
        """
        super().__init__()
        self.mh_attention_layer = MHAttention(query_size, key_size, value_size, hidden_size,
                                              num_heads=num_heads, dropout=dropout)
        self.add_norm_layer1 = Add_and_Norm(hidden_size, dropout=dropout)
        self.ffn_layer = FFN(hidden_size, dropout=dropout)
        self.add_norm_layer2 = Add_and_Norm(hidden_size, dropout=dropout)

    def forward(self, X, valid_len):
        """
        Forward pass of the encoder block.

        :param X (tensor): size (batch_size, num_queries, hidden_size) Input.
        :param valid_len (tensor): size (batch_size, ) Valid length of each sample (sentence).
        :return Z (tensor): size (batch_size, num_queries, hidden_size) Encoder output.
        """
        Y = self.mh_attention_layer(X, X, X, valid_len=valid_len)
        Y = self.add_norm_layer1(Y, X)

        Z = self.ffn_layer(Y)
        Z = self.add_norm_layer2(Z, Y)

        return Z


class DecoderBlock(nn.Module):
    """
    Build the decoder block in the Transformer. It takes in the encoder output as keys and values. And it uses teacher
    forcing technique. The input is a shifted Spanish sentence starting with '<bos>'. The target is the original Spanish
    sentence. To be more specific, it includes 3 sub-layers:
    1. Self-Multihead-Masked attention --> Add & LayerNorm
    2. Multi-head attention with encoder output as keys and values --> Add & LayerNorm
    3. Feed-forward --> Add & LayerNorm
    """
    def __init__(self, query_size, key_size, value_size, hidden_size, num_heads=1, dropout=0.):
        """
        Initialize the decoder block.

        :param query_size (int): The size of the query vector.
        :param key_size (int): The size of the key vector.
        :param value_size (int): The size of the value vector.
        :param hidden_size: The size of the vector which we are going to convert queries/keys/values into.
        :param num_heads (int): The number of attention heads.
        :param dropout (float): The dropout probability.
        """
        super().__init__()
        self.mh_attention_layer1 = MHAttention(query_size, key_size, value_size, hidden_size,
                                               num_heads=num_heads, dropout=dropout)
        self.add_norm_layer1 = Add_and_Norm(hidden_size, dropout=dropout)
        self.mh_attention_layer2 = MHAttention(hidden_size, hidden_size, hidden_size, hidden_size,
                                               num_heads=num_heads, dropout=dropout)
        self.add_norm_layer2 = Add_and_Norm(hidden_size, dropout=dropout)
        self.ffn_layer = FFN(hidden_size, dropout=dropout)
        self.add_norm_layer3 = Add_and_Norm(hidden_size, dropout=dropout)

    def forward(self, X, decoder_valid_len, encoder_output, encoder_valid_len):
        """
        Forward pass of the decoder block. We have to be extra careful with the first self-multihead-masked attention,
        because each query is only allowed to see the key-value pair before it, including itself. So the mask has the shape:
        1 0 0 0 0 0 0 0 0
        1 1 0 0 0 0 0 0 0
        1 1 1 0 0 0 0 0 0
        1 1 1 1 0 0 0 0 0
        ...
        Moreover, notice that the input sentence also has a valid length, so we should also mask the padding tokens. So we
        make the mask more accurate. For instance, if the num_queries = num_keys = 5, but the valid length = 3. Then the mask
        has the following shape:
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 0 0
        1 1 1 0 0

        :param X (tensor): size (batch_size, num_queries, hidden_size) Decoder input.
        :param decoder_valid_len (tensor): size (batch_size, ) Valid length of each sample (Spanish sentence).
        :param encoder_output (tensor): size (batch_size, num_queries, hidden_size) Encoder output.
        :param encoder_valid_len (tensor): size (batch_size, ) Valid length of encoder output.
        :return W (tensor): size (batch_size, num_queries, hidden_size) Decoder output.
        """
        # this valid_len takes both natural masks and decoder_valid_len into account
        # For example, the natural mask is [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        # If the actual lengths are [3, 4], then this returns [[1, 2, 3, 3, 3], [1, 2, 3, 4, 4]], to avoid <pad>
        valid_len = torch.tile(decoder_valid_len.unsqueeze(1), (1, X.shape[1]))

        for idx in range(valid_len.max()):
            valid_len[valid_len[:, idx] > idx, idx] = idx + 1

        Y = self.mh_attention_layer1(X, X, X, valid_len=valid_len)
        Y = self.add_norm_layer1(Y, X)

        Z = self.mh_attention_layer2(Y, encoder_output, encoder_output, valid_len=encoder_valid_len)
        Z = self.add_norm_layer2(Z, Y)

        W = self.ffn_layer(Z)
        W = self.add_norm_layer3(W, Z)

        return W


class Transformer(nn.Module):
    """
    Assemble all the modules to build the Transformer. The main structure is Encoder-Decoder. But before we invoke
    encoder or decoder, we need to convert the one-hot vector into continuous vector, and add positional encoding as
    mentioned before. In the end, we still need a fully connected linear layer to make the output has the size of Spanish
    vocabulary, so that we can obtain a probability distribution over the vocabulary.
    """
    def __init__(self, num_encoders, num_decoders, src_vocab_size, tgt_vocab_size, seq_len, hidden_size, dropout=0.,
                 device=torch.device('cpu')):
        """
        Initialize the Transformer.

        :param num_encoders (int): Number of encoder modules.
        :param num_decoders (int): Number of decoder modules.
        :param src_vocab_size (int): Size of the English vocabulary.
        :param tgt_vocab_size (int): Size of the Spanish vocabulary.
        :param seq_len (int): Sequence length.
        :param hidden_size (int): The size of the vector which we are going to convert queries/keys/values into.
        :param dropout (float): The dropout probability.
        :param device (torch.device): The device on which the model is being trained.
        """
        super().__init__()
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.encoder_embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, hidden_size)
        self.position_encoder = PositionEncoding(hidden_size, seq_len, dropout=dropout, device=device)

        self.encoder_list = []
        for i in range(num_encoders):
            self.encoder_list.append(EncoderBlock(hidden_size, hidden_size, hidden_size, hidden_size,
                                                  num_heads=4, dropout=dropout))
        self.decoder_list = []
        for i in range(num_decoders):
            self.decoder_list.append(DecoderBlock(hidden_size, hidden_size, hidden_size, hidden_size,
                                                  num_heads=4, dropout=dropout))

        self.encoder = nn.Sequential(*self.encoder_list)
        self.decoder = nn.Sequential(*self.decoder_list)

        self.linear_output = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, encoder_in, decoder_in, encoder_valid_len, decoder_valid_len):
        """
        Forward pass of the Transformer.

        :param encoder_in (tensor): size (batch_size, seq_len) The English sentences converted into torch.tensor.
        :param decoder_in (tensor): size (batch_size, seq_len) The Shifted Spanish sentences converted into torch.tensor, beginning with token '<bos>'
        :param encoder_valid_len (int): size (batch_size, ) The actual length of English sentences.
        :param decoder_valid_len (int): size (batch_size, ) The actual length of Spanish sentences.
        :return: decoder_out (tensor): size (batch_size, seq_len, tgt_vocab_size) The output logits of the predicted Spanish tokens.
        """
        encoder_in = self.encoder_embedding(encoder_in)
        decoder_in = self.decoder_embedding(decoder_in)

        encoder_in = self.position_encoder(encoder_in)
        decoder_in = self.position_encoder(decoder_in)

        encoder_out = encoder_in
        for i in range(self.num_encoders):
            encoder_out = self.encoder_list[i](encoder_out, valid_len=encoder_valid_len)

        decoder_out = decoder_in
        for i in range(self.num_decoders):
            decoder_out = self.decoder_list[i](decoder_out, decoder_valid_len, encoder_out, encoder_valid_len)

        return self.linear_output(decoder_out)


def translator(encoder_in, dataset, net, temperature=0.1, device=torch.device('cpu')):
    """
    Use the trained Transformer to translate English sentence to Spanish sentence.

    :param encoder_in (tensor): size (1, seq_length) Input sentence indices.
    :param dataset (Translation_Dataset): Translation dataset instance.
    :param net (Transformer): Transformer instance.
    :param temperature (float): Temperature parameter used to adjust randomness.
    :param device: The device to use.
    :return translation (str): Spanish translation of the input sentence.
    """
    net.eval()
    encoder_in = encoder_in.to(device)
    decoder_in = torch.tensor(dataset.vocab_spa['<bos>'], dtype=torch.long, device=device).reshape(1, 1)
    encoder_valid_len = torch.tensor(encoder_in.shape[1]).reshape(1, )
    decoder_valid_len = torch.tensor([1]).reshape(1, )
    count = 1
    translation = ['<bos>']
    while translation[-1] != '<eos>':
        decoder_out = net(encoder_in, decoder_in, encoder_valid_len, decoder_valid_len)[0, -1, :] / temperature
        probabilities = torch.softmax(decoder_out, dim=0)
        next_token_idx = torch.multinomial(probabilities, 1).item()
        next_token = dataset.inv_vocab_spa[next_token_idx]
        translation.append(next_token)

        decoder_in = torch.cat([decoder_in, torch.tensor(next_token_idx, dtype=torch.long, device=device).reshape(1, 1)], dim=1)
        decoder_valid_len = torch.tensor([count + 1]).reshape(1, )
        count += 1
        if count >= dataset.seq_length * 1:
            break
    return ' '.join(translation)


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
        print(f'This dataset consists of {len(dataset.eng_text)} English-Spanish sentence pairs, each has length {seq_length}')
        print(f'After excluding the uncommon words or punctuations, the vocabulary size (including the punctuations) is {dataset.vocab_eng_size}' +
              f' for English, and {dataset.vocab_spa_size} for Spanish')

    device = torch.device("cuda:0")
    epochs = 30
    num_encoders, num_decoders = 2, 2
    hidden_size = 2048
    transformer = Transformer(num_encoders, num_decoders,
                              dataset.vocab_eng_size, dataset.vocab_spa_size,
                              seq_len=seq_length, hidden_size=hidden_size, dropout=0.3, device=device)
    print('-' * 100)
    try:
        transformer.load_state_dict(torch.load('./data/transformer_weights_cpu.pth'))
        print("Transformer weights detected, loading...")
        loss_list = np.load('./data/loss_transformer.npy').tolist()
    except (RuntimeError, FileNotFoundError) as e:
        print("Transformer weights not found. Creating a new transformer weights...")
        loss_list = []

    transformer = transformer.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=1)  # Ignore the padding tokens when computing loss
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    print('-' * 100)
    time1 = time.time()
    for epoch in range(epochs):
        if epoch % 10 == 0:
            torch.save(transformer.state_dict(), './data/transformer_weights_cpu.pth')
        transformer.train()
        running_loss = 0.0
        batch_num = 0
        total_size = 0
        t_epoch = time.time()
        print(f"############### Epoch {epoch + 1} ###############")
        for encoder_in, decoder_in, decoder_out, src_valid_len, tgt_valid_len in dataloader:
            encoder_in, decoder_in, decoder_out = encoder_in.to(device), decoder_in.to(device), decoder_out.to(device)
            optimizer.zero_grad()

            decoder_out_hat = transformer(encoder_in, decoder_in, src_valid_len, tgt_valid_len)
            loss = loss_fn(decoder_out_hat.reshape(-1, decoder_out_hat.shape[-1]), decoder_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
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
            generated_text = translator(dataset[idx][0].reshape(1, -1), dataset, transformer, temperature=0.5,
                                        device=device)
            print()
            print(f"Original Sentence: {original_text}")
            print(f"Actual Translation: {actual_text}")
            print(f"Test Translation: {generated_text}")

        print(f'One epoch took {time.time() - t_epoch:.2f} seconds')
        loss_list.append(running_loss / total_size)
        print(f"Loss: {loss_list[-1]: .6f}")

    time2 = time.time()
    print('-' * 100)
    print(f"Training time for {epochs} epochs: {time2 - time1: .2f}s")

    loss_hist = np.array(loss_list)
    np.save('./data/loss_transformer.npy', loss_hist)

    transformer = transformer.cpu()
    torch.save(transformer.state_dict(), './data/transformer_weights_cpu.pth')
    print(
        'Net weights and loss are successfully saved to ./data/transformer_weights.pth and ./data/loss_transformer.npy')
    print('Model Structure: ')
    # print(transformer)