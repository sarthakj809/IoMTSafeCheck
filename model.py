import torch
import torch.nn as nn

class ANN_IDS(nn.Module):
    def __init__(self, input_size, num_classes, hidden_dim=64, num_layers=1):
        super(ANN_IDS, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.dropout_post_lstm = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x_reshaped = x.view(-1, features)
        features_extracted = self.feature_extractor(x_reshaped)
        features_extracted = features_extracted.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features_extracted)
        lstm_out = self.dropout_post_lstm(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        final_features = attn_out[:, -1, :]
        output = self.classifier(final_features)
        return output
