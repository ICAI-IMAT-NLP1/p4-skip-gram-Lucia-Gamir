from typing import List, Tuple, Dict, Generator
from collections import Counter
import torch
import random

try:
    from src.utils import tokenize
except ImportError:
    from utils import tokenize


def load_and_preprocess_data(infile: str) -> List[str]:
    """
    Load text data from a file and preprocess it using a tokenize function.

    Args:
        infile (str): The path to the input file containing text data.

    Returns:
        List[str]: A list of preprocessed and tokenized words from the input text.
    """
    with open(infile) as file:
        text = file.read()  # Read the entire file
    
    # Preprocess and tokenize the text
    # TODO
    tokens: List[str] = tokenize(text)

    return tokens

def create_lookup_tables(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create lookup tables for vocabulary.

    Args:
        words: A list of words from which to create vocabulary.

    Returns:
        A tuple containing two dictionaries. The first dictionary maps words to integers (vocab_to_int),
        and the second maps integers to words (int_to_vocab).
    """
    # TODO
    """
    Example:
    words = ["a", "b", "r", "b", "b"]
    word_counts = {'b': 3, 'a': 1, 'r': 1}
    sorted_vocab = ['b', 'a', 'r']
    enumerate to have int_to_vocab and vocab_to_int
    """
    word_counts: Counter = Counter(words)
    # Sorting the words from most to least frequent in text occurrence.
    sorted_vocab: List[int] = sorted(set(words), key=lambda x: word_counts[x], reverse=True)
    
    # Create int_to_vocab and vocab_to_int dictionaries.
    int_to_vocab: Dict[int, str] = {number:word for number,word in enumerate(sorted_vocab)}
    vocab_to_int: Dict[str, int] = {int_to_vocab[number]: number for number in int_to_vocab}

    return vocab_to_int, int_to_vocab


def subsample_words(words: List[str], vocab_to_int: Dict[str, int], threshold: float = 1e-5) -> Tuple[List[int], Dict[str, float]]:
    """
    Perform subsampling on a list of word integers using PyTorch, aiming to reduce the 
    presence of frequent words according to Mikolov's subsampling technique. This method 
    calculates the probability of keeping each word in the dataset based on its frequency, 
    with more frequent words having a higher chance of being discarded. The process helps 
    in balancing the word distribution, potentially leading to faster training and better 
    representations by focusing more on less frequent words.
    
    Args:
        words (list): List of words to be subsampled.
        vocab_to_int (dict): Dictionary mapping words to unique integers.
        threshold (float): Threshold parameter controlling the extent of subsampling.

        
    Returns:
        List[int]: A list of integers representing the subsampled words, where some high-frequency words may be removed.
        Dict[str, float]: Dictionary associating each word with its frequency.
    """
    # TODO
    """
    Example:
    words = ["a", "b"...]
    int_words = [1,2]

    freqs = {"a":80, "b":2}
    words_tensor = [1,2]
    probabilities = [high, low]
    mask = [false, true]
    train_words_tensor = [2]
    """
    # Convert words to integers
    int_words: List[int] = [vocab_to_int[word] for word in set(words)]
    
    # Compute frequencies of the words
    freqs: Dict[str, float] = Counter(words)

    words_int = [vocab_to_int[word] for word in list(freqs.keys())]
    words_tensor = torch.tensor(words_int)
    freqs_tensor: torch.tensor = torch.tensor(list(freqs.values())) 
    freqs_tensor = freqs_tensor / freqs_tensor.shape[0]

    probabilities = 1 - torch.sqrt(threshold/freqs_tensor)
    mask = probabilities <= torch.rand(probabilities.shape[0])
    train_words_tensor = words_tensor[mask]

    train_words: List[int] = train_words_tensor.tolist()

    return train_words, freqs

def get_target(words: List[str], idx: int, window_size: int = 5) -> List[str]:
    """
    Get a list of words within a window around a specified index in a sentence.

    Args:
        words (List[str]): The list of words from which context words will be selected.
        idx (int): The index of the target word.
        window_size (int): The maximum window size for context words selection.

    Returns:
        List[str]: A list of words selected randomly within the window around the target word.
    """
    # TODO
    r = random.randint(1, window_size+1)

    target_words: List[str] = words[idx-r:idx] + words[idx+1:idx+r+1]

    return target_words

def get_batches(words: List[int], batch_size: int, window_size: int = 5) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Generate batches of word pairs for training.

    This function creates a generator that yields tuples of (inputs, targets),
    where each input is a word, and targets are context words within a specified
    window size around the input word. This process is repeated for each word in
    the batch, ensuring only full batches are produced.

    Args:
        words: A list of integer-encoded words from the dataset.
        batch_size: The number of words in each batch.
        window_size: The size of the context window from which to draw context words.

    Yields:
        A tuple of two lists:
        - The first list contains input words (repeated for each of their context words).
        - The second list contains the corresponding target context words.
    """

    # TODO
    for idx in range(0, len(words), batch_size):
        inputs, targets = [], []
        batch = words[idx:idx+batch_size]

        for i in range(len(batch)):
            input_word = batch[i]

            if i < window_size:
                previous_context_words = batch[0:i]
            else:
                previous_context_words = batch[i-window_size:i]
            
            if i > (len(batch)-window_size):
                following_context_words = batch[i+1:len(batch)]
            else:
                following_context_words = batch[i+1:i+1+window_size]
        
            context_words = previous_context_words + following_context_words

            inputs += [input_word]*len(context_words)
            targets += context_words 


        yield inputs, targets

def cosine_similarity(embedding: torch.nn.Embedding, valid_size: int = 16, valid_window: int = 100, device: str = 'cpu'):
    """Calculates the cosine similarity of validation words with words in the embedding matrix.

    This function calculates the cosine similarity between some random words and
    embedding vectors. Through the similarities, it identifies words that are
    close to the randomly selected words.

    Args:
        embedding: A PyTorch Embedding module.
        valid_size: The number of random words to evaluate.
        valid_window: The range of word indices to consider for the random selection.
        device: The device (CPU or GPU) where the tensors will be allocated.

    Returns:
        A tuple containing the indices of valid examples and their cosine similarities with
        the embedding vectors.

    Note:
        sim = (a . b) / |a||b| where `a` and `b` are embedding vectors.
    """

    # TODO
    # Take valid_size indexes in a range between 0 and valid_window
    valid_examples = torch.randint(0, valid_window, (valid_size,), device=device)

    # Valid embeddings
    valid_embeddings = embedding(valid_examples)  # (valid_size, embedding_dim)
    
    # All the embeddings we have
    all_embeddings = embedding.weight  # (vocab_size, embedding_dim)

    # Compute the norm of the vectors
    valid_norms = torch.norm(valid_embeddings, dim=1, keepdim=True)  # (valid_size, 1)
    all_norms = torch.norm(all_embeddings, dim=1, keepdim=True)  # (vocab_size, 1)

    # Compute similarities with the formula
    similarities = (valid_embeddings @ all_embeddings.T) / (valid_norms * all_norms.T)  # (valid_size, vocab_size)

    return valid_examples, similarities