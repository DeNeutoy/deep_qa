import codecs
import gzip
import logging

import numpy
from keras.layers import Embedding
from ..data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PretrainedEmbeddings:
    @staticmethod
    def initialize_random_matrix(shape, seed=1337):
        # TODO(matt): we now already set the random seed, in run_solver.py.  This should be
        # changed.
        numpy_rng = numpy.random.RandomState(seed)
        return numpy_rng.uniform(size=shape, low=0.05, high=-0.05)

    @staticmethod
    def get_embedding_layer(embeddings_filename: str,
                            vocab: Vocabulary,
                            namespace: str="tokens",
                            trainable=False,
                            log_misses=False,
                            name="pretrained_embedding"):
        """
        Reads a pre-trained embedding file and generates a Keras Embedding layer that has weights
        initialized to the pre-trained embeddings.  The Embedding layer can either be trainable or
        not.

        We use the DataIndexer to map from the word strings in the embeddings file to the indices
        that we need, and to know which words from the embeddings file we can safely ignore.  If we
        come across a word in DataIndexer that does not show up with the embeddings file, we give
        it a zero vector.

        The embeddings file is assumed to be gzipped, formatted as [word] [dim 1] [dim 2] ...
        """
        words_to_keep = set(vocab.tokens_in_namespace(namespace))
        vocab_size = vocab.get_vocab_size(namespace)
        embeddings = {}
        embedding_dim = None

        # TODO(matt): make this a parameter
        embedding_misses_filename = 'embedding_misses.txt'

        # First we read the embeddings from the file, only keeping vectors for the words we need.
        logger.info("Reading embeddings from file")
        with gzip.open(embeddings_filename, 'rb') as embeddings_file:
            for line in embeddings_file:
                fields = line.decode('utf-8').strip().split(' ')
                if embedding_dim is None:
                    embedding_dim = len(fields) - 1
                    assert embedding_dim > 1, "Found embedding size of 1; do you have a header?"
                else:
                    if len(fields) - 1 != embedding_dim:
                        # Sometimes there are funny unicode parsing problems that lead to different
                        # fields lengths (e.g., a word with a unicode space character that splits
                        # into more than one column).  We skip those lines.  Note that if you have
                        # some kind of long header, this could result in all of your lines getting
                        # skipped.  It's hard to check for that here; you just have to look in the
                        # embedding_misses_file and at the model summary to make sure things look
                        # like they are supposed to.
                        continue
                word = fields[0]
                if word in words_to_keep:
                    vector = numpy.asarray(fields[1:], dtype='float32')
                    embeddings[word] = vector

        # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
        # then filling in the word vectors we just read.
        logger.info("Initializing pre-trained embedding layer")
        if log_misses:
            logger.info("Logging embedding misses to %s", embedding_misses_filename)
            embedding_misses_file = codecs.open(embedding_misses_filename, 'w', 'utf-8')
        embedding_matrix = PretrainedEmbeddings.initialize_random_matrix((vocab_size, embedding_dim))

        # Depending on whether the namespace has PAD and UNK tokens, we start at different indices,
        # because you shouldn't have pretrained embeddings for PAD or UNK.
        start_index = 0 if namespace.startswith("*") else 2
        for i in range(start_index, vocab_size):
            word = vocab.get_token_from_index(i, namespace)

            # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
            # so the word has a random initialization.
            if word in embeddings:
                embedding_matrix[i] = embeddings[word]
            elif log_misses:
                print(word, file=embedding_misses_file)

        if log_misses:
            embedding_misses_file.close()

        # The weight matrix is initialized, so we construct and return the actual Embedding layer.
        return Embedding(input_dim=vocab_size,
                         output_dim=embedding_dim,
                         mask_zero=True,
                         weights=[embedding_matrix],
                         trainable=trainable,
                         name=name)
