from transformers import AutoModel
import torch

class BertWithExtraFeature(torch.nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=768, extra_feature_size=1):
        super(BertWithExtraFeature, self).__init__()
        # Load the pre-trained BERT model
        self.bert = AutoModel.from_pretrained(bert_model_name, cache_dir = '/Data')
        self.hidden_size = hidden_size
        self.extra_feature_size = extra_feature_size
        
        self.dropout = torch.nn.Dropout(p=0.5)
        # Fully connected layer to combine BERT output and extra feature
        self.fc = torch.nn.Linear(self.hidden_size + self.extra_feature_size, 2)
    
    def forward(self, input_ids, attention_mask, extra_feature):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with token IDs.
            attention_mask: Tensor of shape (batch_size, seq_len) for masking attention.
            extra_feature: Tensor of shape (batch_size, 1) with the additional feature.

        Returns:
            Logits for binary classification.
        """
        # Get BERT output (pooled output)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # Shape: (batch_size, hidden_size)

        pooled_output = self.dropout(pooled_output)
        
        # Concatenate the pooled output with the extra feature
        combined_input = torch.cat((pooled_output, extra_feature), dim=1)  # Shape: (batch_size, hidden_size + extra_feature_size)
        
        # Pass through the fully connected layer
        logits = self.fc(combined_input)  # Shape: (batch_size, 1)
        
        return logits