import os,glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class InferenceDataset(Dataset):
    def __init__(self, ids, texts, tokenizer, max_length=128):
        self.ids = ids
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        text = self.texts[idx]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'id': self.ids[idx],
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }

# Define the model architecture
class BertRegressionModel(nn.Module):
    def __init__(self, bert_model_name, output_size):
        super(BertRegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regression_head = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Get the [CLS] token output
        cls_output = outputs.last_hidden_state[:, 0, :]
        predictions = self.regression_head(cls_output)
        return predictions

def predict(model, data_loader):
        model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask)
                all_predictions.extend(outputs.cpu().numpy())

        return all_predictions


if __name__=="__main__":
    test_file_list=glob.glob("./testing_data/*.csv")

    for test_file in test_file_list:
        # Load test data
        test_data = pd.read_csv(test_file)

        # Create tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        # Create dataset for inference
        inference_dataset = InferenceDataset(test_data['ID'], test_data['Text'], tokenizer)

        ## Valence

        # Load the pre-trained model
        valence_model = BertRegressionModel("bert-base-chinese", output_size=2).to(device)
        arousal_model = BertRegressionModel("bert-base-chinese", output_size=2).to(device)
        valence_model.load_state_dict(torch.load(f"./model_weight/bert_regression_model_valence/Valence_model_state_{0.012}.pth"))
        arousal_model.load_state_dict(torch.load(f"./model_weight/bert_regression_model_valence/Arousal_model_state_{0.024}.pth"))

        inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

        # Get valence predictions
        valence_predictions = predict(valence_model, inference_loader)
        # Considering only 'Valence_Mean'
        val_predictions = [pred[0] for pred in valence_predictions]

        # Save valence predictions to CSV
        predictions_df = pd.DataFrame(val_predictions, columns=['Valence'])
        predictions_df['ID'] = test_data['ID']
        predictions_df = predictions_df[['ID', 'Valence']]
        predictions_df.to_csv(f"./predict_result/valence_predictions.csv", index=False)

        # Get valence predictions
        arousal_predictions = predict(arousal_model, inference_loader)

        aro_predictions = [pred[0] for pred in arousal_predictions]  # Considering only 'Valence_Mean'

        # Save valence predictions to CSV
        predictions_df = pd.DataFrame(aro_predictions, columns=['Arousal'])
        predictions_df['ID'] = test_data['ID']
        predictions_df = predictions_df[['ID', 'Arousal']]
        predictions_df.to_csv(f"./predict_result/arousal_predictions.csv", index=False)

        # Read the valence predictions CSV
        valence_predictions_df = pd.read_csv(f"./predict_result/valence_predictions.csv")

        # Read the arousal predictions CSV
        arousal_predictions_df = pd.read_csv(f"./predict_result/arousal_predictions.csv")

        # Merge the two DataFrames based on the 'ID' column
        submission_df = pd.merge(valence_predictions_df, arousal_predictions_df, on='ID')

        # Save the combined DataFrame to a submission CSV file
        submission_df.to_csv("./predict_result/submission.csv", index=False)
