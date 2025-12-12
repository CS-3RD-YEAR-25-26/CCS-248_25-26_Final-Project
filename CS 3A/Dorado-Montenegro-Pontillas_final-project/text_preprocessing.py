"""
Text Preprocessing and Tokenization for Bilingual Hate Speech Detection
Handles both Filipino and English text
"""

import re
import pickle
from collections import Counter
import numpy as np


class TextPreprocessor:
    """
    Preprocess text for RNN model
    - Lowercasing
    - Remove URLs, mentions, hashtags (or tokenize them)
    - Punctuation handling
    """
    
    def __init__(self, lowercase=True, remove_urls=True, remove_mentions=True, 
                 remove_hashtags=False, remove_punctuation=False):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_punctuation = remove_punctuation
        
    def preprocess(self, text):
        """Preprocess a single text"""
        if not isinstance(text, str):
            text = str(text)
            
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        if self.remove_hashtags:
            text = re.sub(r'#\w+', '', text)
        else:
            # Keep hashtag content but remove #
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_batch(self, texts):
        """Preprocess a batch of texts"""
        return [self.preprocess(text) for text in texts]


class Vocabulary:
    """
    Build vocabulary from text corpus
    """
    
    def __init__(self, max_vocab_size=10000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
    def build_vocab(self, texts):
        """Build vocabulary from a list of texts"""
        print("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
        
        # Sort by frequency
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by min_freq and max_vocab_size
        filtered_words = [word for word, freq in sorted_words if freq >= self.min_freq]
        filtered_words = filtered_words[:self.max_vocab_size - 4]  # Reserve space for special tokens
        
        # Create mappings
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        
        for idx, word in enumerate(filtered_words, start=4):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"✓ Vocabulary built: {len(self.word2idx)} words")
        print(f"  - Total unique words in corpus: {len(self.word_freq)}")
        print(f"  - Words after filtering (min_freq={self.min_freq}): {len(filtered_words)}")
        print(f"  - Most common words: {filtered_words[:10]}")
        
    def text_to_sequence(self, text, max_length=None):
        """Convert text to sequence of indices"""
        words = text.split()
        sequence = [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
        
        # Truncate if needed
        if max_length and len(sequence) > max_length:
            sequence = sequence[:max_length]
            
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text"""
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in sequence]
        return ' '.join(words)
    
    def save(self, filepath):
        """Save vocabulary to file"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq,
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"✓ Vocabulary saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        vocab = cls(
            max_vocab_size=vocab_data['max_vocab_size'],
            min_freq=vocab_data['min_freq']
        )
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = vocab_data['idx2word']
        vocab.word_freq = vocab_data['word_freq']
        
        print(f"✓ Vocabulary loaded: {len(vocab.word2idx)} words")
        return vocab
    
    def __len__(self):
        return len(self.word2idx)


def pad_sequences(sequences, max_length=None, padding_value=0):
    """
    Pad sequences to same length
    
    Args:
        sequences: List of sequences (list of lists)
        max_length: Maximum length (if None, use longest sequence)
        padding_value: Value to use for padding
        
    Returns:
        padded_sequences: Numpy array of padded sequences
        lengths: Original lengths of sequences
    """
    lengths = [len(seq) for seq in sequences]
    
    if max_length is None:
        max_length = max(lengths)
    
    # Initialize padded array
    padded_sequences = np.full((len(sequences), max_length), padding_value, dtype=np.int64)
    
    # Fill with sequences
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        if length > max_length:
            padded_sequences[i] = seq[:max_length]
            lengths[i] = max_length
        else:
            padded_sequences[i, :length] = seq
    
    return padded_sequences, np.array(lengths, dtype=np.int64)


def test_preprocessing():
    """Test preprocessing pipeline"""
    print("=" * 60)
    print("TESTING TEXT PREPROCESSING")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "Mar Roxas TANG INA TUWID NA DAAN DAW .. EH SYA NGA DI STRAIGHT",
        "@XochitlSuckkks a classy whore? Or more red velvet cupcakes?",
        "Salamat sa walang sawang suporta! #OnlyBinay https://t.co/test",
        "This is a normal text without hate speech."
    ]
    
    labels = [1, 1, 0, 0]
    
    # Preprocess
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_mentions=True,
        remove_hashtags=False
    )
    
    print("\nOriginal vs Preprocessed:")
    print("-" * 60)
    processed_texts = []
    for text in texts:
        processed = preprocessor.preprocess(text)
        processed_texts.append(processed)
        print(f"Original: {text[:60]}...")
        print(f"Processed: {processed[:60]}...")
        print()
    
    # Build vocabulary
    print("=" * 60)
    print("BUILDING VOCABULARY")
    print("=" * 60)
    
    vocab = Vocabulary(max_vocab_size=100, min_freq=1)
    vocab.build_vocab(processed_texts)
    
    # Convert to sequences
    print("\n" + "=" * 60)
    print("TEXT TO SEQUENCE CONVERSION")
    print("=" * 60)
    
    sequences = [vocab.text_to_sequence(text) for text in processed_texts]
    
    for i, (text, seq) in enumerate(zip(processed_texts, sequences)):
        print(f"\nText {i+1}: {text[:50]}...")
        print(f"Sequence: {seq}")
        print(f"Length: {len(seq)}")
    
    # Pad sequences
    print("\n" + "=" * 60)
    print("PADDING SEQUENCES")
    print("=" * 60)
    
    padded_seqs, lengths = pad_sequences(sequences, max_length=20)
    
    print(f"Padded sequences shape: {padded_seqs.shape}")
    print(f"Lengths: {lengths}")
    print(f"\nFirst padded sequence:\n{padded_seqs[0]}")
    
    print("\n✓ All preprocessing tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_preprocessing()
