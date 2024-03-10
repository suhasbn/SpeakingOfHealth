import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
#import torchaudio
from torch.utils.data import DataLoader, Dataset
from msclap import CLAP
import os
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

required_subjects = {
    1: [1, 12, 14, 16, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 3, 30, 31, 33, 34, 35, 36, 37, 4, 40, 5, 6, 7, 8],
    2: [1, 14, 16, 17, 19, 21, 22, 25, 26, 28, 29, 3, 31, 33, 34, 35, 36, 37, 4, 40, 6, 7, 8],
    3: [1, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26, 28, 29, 31, 33, 36, 4, 40, 8]
}

# Load sentence embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("Sentence embedding model loaded successfully")
class AttentionModule(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, embeddings):
        scores = torch.matmul(embeddings, self.query)  # Assuming [seq_length, embedding_dim]
        attention_weights = F.softmax(scores, dim=0)
        weighted_sum = torch.einsum('n,nm->m', attention_weights, embeddings)
        return weighted_sum

class SessionDataset(Dataset):
    def __init__(self, csv_path, audio_base_folder, summary_file, clap_model, df_for_training, embedding_dim, scenario, subset):
        self.data = pd.read_csv(csv_path)
        self.audio_base_folder = audio_base_folder
        self.clap_model = clap_model
        self.df_for_training = df_for_training
        self.embedding_dim = embedding_dim
        self.scenario = scenario
        self.subset = subset  # Use subset to filter data

        # Map scenarios to summary files
        summary_file_paths = {
            'no_summary': None,  # No summary data
            'only_phi2': '/dgxhome/sxb701/Amanda_Speech_Transcript_Data/Codes/classified_session_transcriptions_with_speakers_phi2_all.csv',
            'only_meditron': '/dgxhome/sxb701/Amanda_Speech_Transcript_Data/Codes/classified_session_transcriptions_with_speakers_meditron_all.csv',
            'only_llama2': '/dgxhome/sxb701/Amanda_Speech_Transcript_Data/Codes/classified_session_transcriptions_with_speakers_meditron_all.csv',
        }

        summary_file = summary_file_paths.get(scenario)
        self.summaries = pd.read_csv(summary_file) if summary_file else None

        
        
        # Add a step to extract subject IDs from filenames
        self.data['SubjectID'] = self.data['Filename'].apply(self.extract_subject_id)
        
        # Filter data based on subset type
        self.subset = subset
        if self.subset == 'train':
            self.subject_ids = [1, 14, 28, 21, 17, 35, 40, 4, 31, 16, 18, 25, 7, 6, 30, 24, 2]
        elif self.subset == 'val':
            self.subject_ids = [29, 8, 19, 34, 12]
        elif self.subset == 'test':
            self.subject_ids = [33, 26, 3, 36, 22, 23, 37, 5, 27]
        else:
            raise ValueError("Invalid subset type. Choose from 'train', 'val', or 'test'.")
        
        # Assuming there's a column in your CSV that indicates the subject ID for each session
        # Adjust the column name 'SubjectID' according to your actual CSV format
        self.data = self.data[self.data['SubjectID'].isin(self.subject_ids)]
        
    def extract_subject_id(self, filename):
        # Regex to find digits following 'P' and before the first dot '.'
        match = re.search(r'P(\d+).', filename)
        if match:
            return int(match.group(1))  # Return the subject ID as integer
        else:
            return None  # or raise an error, depending on how you want to handle files without a valid format

    def __len__(self):
        return len(self.data['Filename'].unique())

    def build_audio_path(self, filename):
        interview_part = filename.split('_')[0]
        audio_path = os.path.join(self.audio_base_folder, f"Interview{interview_part[1:]}", filename)
        print(f"Loading audio from: {audio_path}")  # Print the audio file path
        return audio_path
    
    def get_summary_embedding(self, filename):

        # If no summaries are provided, return a zero embedding immediately
        if self.summaries is None:
            return torch.zeros(self.embedding_dim, device=device)
    # Convert filename format from 'I1_P12.24.05.18_16k.wav' to 'I1P12.24.05.18.docx'
    # to match the LLM summary data format
        session_id = filename.split('_')[0] + filename.split('_')[1].split('.')[0]  # e.g., 'I1P12'
        # Find the corresponding LLM summary
        summary_row = self.summaries[self.summaries['Filename'].str.contains(session_id, regex=False)]
        if not summary_row.empty:
            summary_text = summary_row['Descriptive_Sentence'].iloc[0]
            # Process the summary text through CLAP to get the embedding
            summary_embedding = self.clap_model.get_text_embeddings([summary_text])[0]  # Adjust indexing based on output
            print(f"Summary embedding shape for {filename}: {summary_embedding.shape}")
            return summary_embedding
        else:
            # Return a zero embedding if no summary is available
            return torch.zeros(self.embedding_dim, device=device)

    def __getitem__(self, idx):
        session_info = self.data.iloc[idx]  # Directly access the row for this session
        session_id = session_info['id']
        survey_no = session_info['survey.no']
        filename = session_info['Filename']  # Assuming 'Filename' column exists
        embeddings_list = []

        # Use df_for_training passed during the dataset initialization
        session_labels = self.df_for_training[(self.df_for_training['id'] == session_id) & (self.df_for_training['survey.no'] == survey_no)]
        if not session_labels.empty:
            cosc_labels = session_labels[['how.often.1', 'what.extent.1', 'how.often.2', 'what.extent.2', 'paMIN.1', 'paMIN.2', 'rai.1', 'rai.2', 'dep', 'anx', 'stress', 'aqol.total']].values.flatten()


        audio_path = self.build_audio_path(filename)
        audio_embedding = self.clap_model.get_audio_embeddings([audio_path])[0]
        #print(f"Audio embedding shape for {filename}: {audio_embedding.shape}")
        embeddings_list.append(audio_embedding)

        text_embedding = self.clap_model.get_text_embeddings([session_info['Transcript']])[0]
        embeddings_list.append(text_embedding)

        summary_embedding = self.get_summary_embedding(filename)  # Use filename for summary embedding
        embeddings_list.append(summary_embedding)

        embeddings_tensor = torch.stack(embeddings_list)
        attention_module = AttentionModule(embeddings_tensor.shape[-1]).to(device)
        aggregated_embedding = attention_module(embeddings_tensor)

        return aggregated_embedding, torch.tensor(cosc_labels, dtype=torch.float)

# Initialize CLAP model (adjust path and settings as needed)
clap_model = CLAP(version='2023', use_cuda=device.type == 'cuda')
print("CLAP model loaded successfully")

# Example dataset and dataloader initialization
df_for_training = pd.read_csv('/dgxhome/sxb701/Amanda_Speech_Transcript_Data/Codes/SCORED_short_useful_imputed.csv')
csv_path = '/dgxhome/sxb701/Amanda_Speech_Transcript_Data/Codes/updated_transcript_selected_with_id_survey_no.csv'
audio_base_folder = '/dgxhome/sxb701/Amanda_Speech_Transcript_Data/'

# Assuming you have a function or loop to handle this
def create_datasets_and_loaders(scenario):
    summary_file = summary_paths[scenario]
    train_dataset = SessionDataset(csv_path, audio_base_folder, summary_file, clap_model, df_for_training, 1024, scenario, 'train')
    val_dataset = SessionDataset(csv_path, audio_base_folder, summary_file, clap_model, df_for_training, 1024, scenario, 'val')
    test_dataset = SessionDataset(csv_path, audio_base_folder, summary_file, clap_model, df_for_training, 1024, scenario, 'test')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, val_loader, test_loader

class MultimodalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_encoder_layers, num_cos_labels):
        super(MultimodalTransformer, self).__init__()
        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        # Output heads for COSC and MCMLC tasks
        self.cos_head = nn.Linear(input_dim, num_cos_labels)  # COSC output layer

    def forward(self, x, mask=None):
        # x: batch_size x seq_len x input_dim
        # Transform x to match transformer input dimensions: seq_len x batch_size x input_dim
        x = x.unsqueeze(1)  # Now x has shape [batch_size, 1, embedding_dim]
        x = x.permute(1, 0, 2)
        
        # Passing the input through the Transformer encoder
        transformed = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Reverting dimensions to batch_size x seq_len x input_dim
        transformed = transformed.permute(1, 0, 2)
        
        # Pooling over the seq_len dimension to get a single representation for the session
        pooled = torch.mean(transformed, dim=1)
        
        # Getting output for both tasks
        cosc_output = self.cos_head(pooled)  # Predictions for COSC

        return cosc_output

def adaptive_huber_loss(cosc_preds, cosc_targets, weights, deltas):
    """
    Calculate the Adaptive Huber Loss for predictions and targets, with variable-specific
    thresholds (deltas) and importance weights.
    """
    abs_errors = torch.abs(cosc_preds - cosc_targets)
    quadratic_part = torch.where(abs_errors < deltas, 0.5 * abs_errors ** 2, deltas * (abs_errors - 0.5 * deltas))
    weighted_loss = torch.sum(weights * quadratic_part, dim=1)  # Apply weights and sum over variables
    mean_loss = torch.mean(weighted_loss)  # Average over all examples in the batch
    return mean_loss

# (how.often.1       0.250919
#  what.extent.1     0.236535
#  how.often.2       0.163667
#  what.extent.2     0.132493
#  paMIN.1          19.051768
#  paMIN.2           8.715625
#  rai.1             0.897858
#  rai.2             0.546192
#  dep               1.188696
#  anx               1.102142
#  stress            1.501584
#  aqol.total        2.042644
#  dtype: float64,
#  how.often.1      3.985354
#  what.extent.1    4.227699
#  how.often.2      6.109950
#  what.extent.2    7.547585
#  paMIN.1          0.052489
#  paMIN.2          0.114736
#  rai.1            1.113761
#  rai.2            1.830858
#  dep              0.841258
#  anx              0.907324
#  stress           0.665963
#  aqol.total       0.489562


# Adjustments to the training loop
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, deltas, weights, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        # Training phase
        for batch in train_dataloader:
            inputs, cosc_targets = batch
            inputs, cosc_targets = inputs.to(device), cosc_targets.to(device)

            optimizer.zero_grad()
            cosc_preds = model(inputs)
            loss = adaptive_huber_loss(cosc_preds, cosc_targets, weights, deltas)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, cosc_targets = batch
                inputs, cosc_targets = inputs.to(device), cosc_targets.to(device)
                cosc_preds = model(inputs)
                loss = adaptive_huber_loss(cosc_preds, cosc_targets, weights, deltas)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f'Epoch {epoch+1}, Total Loss: {total_loss}, Val Loss: {val_loss}')
        scheduler.step(val_loss)  # Adjust the learning rate based on validation loss

#Evaluation
def evaluate_model_per_variable(model, dataloader, device):
    model.eval()
    total_loss = 0
    variable_losses = torch.zeros(12, device=device)  # Assuming 12 variables
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, cosc_targets = batch
            inputs, cosc_targets = inputs.to(device), cosc_targets.to(device)
            cosc_preds = model(inputs)

            # Calculate MSE for each variable
            mse_loss = F.mse_loss(cosc_preds, cosc_targets, reduction='none').mean(0)
            variable_losses += mse_loss
            count += 1

    variable_losses /= count
    return variable_losses.cpu().numpy()  # Return numpy array for easier handling

deltas_values = [0.25, 0.23, 0.16, 0.13, 19.05, 8.71, 0.89, 0.54, 1.18, 1.10, 1.50, 2.04]
weights_values = [3.98, 4.22, 6.10, 7.54, 0.05, 0.11, 1.11, 1.83, 0.84, 0.90, 0.66, 0.48]

# Convert list values to tensors
deltas = torch.tensor(deltas_values, dtype=torch.float32, device=device)
weights = torch.tensor(weights_values, dtype=torch.float32, device=device)

# Assuming you have dataset variants for each scenario
summary_paths = {
    'no_summary': None,  # No summary data
    'only_phi2': '/dgxhome/sxb701/Amanda_Speech_Transcript_Data/Codes/classified_session_transcriptions_with_speakers_phi2_all.csv',
    'only_meditron': '/dgxhome/sxb701/Amanda_Speech_Transcript_Data/Codes/classified_session_transcriptions_with_speakers_meditron_all.csv',
    'only_llama2': '/dgxhome/sxb701/Amanda_Speech_Transcript_Data/Codes/classified_session_transcriptions_with_speakers_meditron_all.csv',
    #'all': '/path/to/combined/summary.csv',  # Assuming you have a way to combine summaries
}

# # Initialize dataset instances for each scenario
# datasets = {}
# for scenario, summary_file in summary_paths.items():
#     datasets[scenario] = SessionDataset(
#         csv_path=csv_path,
#         audio_base_folder=audio_base_folder,
#         summary_file=summary_file,  # Pass the specific summary file path for the scenario
#         clap_model=clap_model,
#         df_for_training=df_for_training,
#         embedding_dim=1024,
#         subset=scenario  # The subset parameter could be repurposed to label the scenario
#     )

# Initialize the model and optimizer
model = MultimodalTransformer(input_dim=1024, num_heads=8, num_encoder_layers=6, num_cos_labels=12).to(device)  # Adjust num_cos_labels if needed
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Iterate over scenarios

scenarios = ['no_summary', 'only_phi2', 'only_meditron', 'only_llama2']
for scenario in scenarios:
    # Create DataLoaders for the current scenario
    train_loader, val_loader, test_loader = create_datasets_and_loaders(scenario)

    # Initialize a fresh model for each scenario
    model = MultimodalTransformer(input_dim=1024, num_heads=8, num_encoder_layers=6, num_cos_labels=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
    
    # Train the model with the scenario-specific train and val DataLoaders
    train_model(model, train_loader, val_loader, optimizer, scheduler, deltas, weights, num_epochs=100)

    # Save the trained model specific to the scenario
    torch.save(model.state_dict(), f'model_{scenario}.pth')

    # Load the scenario-specific trained model for evaluation
    model.load_state_dict(torch.load(f'model_{scenario}.pth'))
    variable_losses = evaluate_model_per_variable(model, test_loader, device)  # Use the correct test_loader
    print(f"Results for {scenario}:")
    for i, loss in enumerate(variable_losses):
        print(f"Variable {i+1} MSE: {loss}")