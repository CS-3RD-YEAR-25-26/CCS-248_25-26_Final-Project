"""
BiLSTM Model for Bilingual Hate Speech Detection
Supports both Filipino and English text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMHateSpeechClassifier(nn.Module):
    """
    Bidirectional LSTM for hate speech classification
    
    Architecture:
    Input Text → Embedding → Bidirectional LSTM → Dropout → Dense → Output
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, 
                 num_layers=2, dropout=0.3, pad_idx=0):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            pad_idx: Index for padding token
        """
        super(BiLSTMHateSpeechClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        # *2 because bidirectional (forward + backward)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, text, text_lengths):
        """
        Forward pass
        
        Args:
            text: Tensor of token indices [batch_size, seq_len]
            text_lengths: Actual lengths of sequences [batch_size]
            
        Returns:
            predictions: Tensor of predictions [batch_size, 1]
        """
        # Embedding: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(text))
        
        # Pack padded sequences for efficient LSTM processing
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM: [batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, hidden_dim*2]
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Concatenate final forward and backward hidden states
        # hidden: [num_layers*2, batch_size, hidden_dim]
        # Get last layer: [-2] is forward, [-1] is backward
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # Apply dropout
        hidden_concat = self.dropout(hidden_concat)
        
        # Fully connected layers
        fc1_output = F.relu(self.fc1(hidden_concat))
        fc1_output = self.dropout(fc1_output)
        
        # Output: [batch_size, 1]
        predictions = self.fc2(fc1_output)
        
        return predictions
    
    def predict(self, text, text_lengths):
        """
        Predict with sigmoid activation
        
        Returns:
            probabilities: Probability of hate speech [batch_size]
        """
        logits = self.forward(text, text_lengths)
        probabilities = torch.sigmoid(logits).squeeze(1)
        return probabilities


class SimpleBiLSTM(nn.Module):
    """
    Simpler BiLSTM model (alternative architecture)
    Good for quick experiments
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, dropout=0.3, pad_idx=0):
        super(SimpleBiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Concatenate last hidden states
        hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden_concat = self.dropout(hidden_concat)
        
        # Output
        output = self.fc(hidden_concat)
        return output
    
    def predict(self, text, text_lengths):
        logits = self.forward(text, text_lengths)
        probabilities = torch.sigmoid(logits).squeeze(1)
        return probabilities


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model initialization and forward pass"""
    print("=" * 60)
    print("TESTING BiLSTM MODEL")
    print("=" * 60)
    
    # Hyperparameters
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 128
    batch_size = 32
    seq_len = 50
    
    # Initialize model
    model = BiLSTMHateSpeechClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3
    )
    
    print(f"Model: BiLSTMHateSpeechClassifier")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    print(f"\n" + "=" * 60)
    print("TESTING FORWARD PASS")
    print("=" * 60)
    
    # Create dummy input
    dummy_text = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_lengths = torch.randint(10, seq_len, (batch_size,))
    
    print(f"Input shape: {dummy_text.shape}")
    print(f"Text lengths: {dummy_lengths.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_text, dummy_lengths)
        predictions = model.predict(dummy_text, dummy_lengths)
    
    print(f"\nOutput (logits) shape: {output.shape}")
    print(f"Predictions (probabilities) shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5].numpy()}")
    
    print("\n✓ Model test passed!")
    print("=" * 60)
    
    # Test simple model
    print("\n" + "=" * 60)
    print("TESTING SIMPLE BiLSTM MODEL")
    print("=" * 60)
    
    simple_model = SimpleBiLSTM(vocab_size=vocab_size, embedding_dim=100, hidden_dim=64)
    print(f"Trainable parameters: {count_parameters(simple_model):,}")
    
    with torch.no_grad():
        output = simple_model(dummy_text, dummy_lengths)
    print(f"Output shape: {output.shape}")
    print("✓ Simple model test passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_model()
