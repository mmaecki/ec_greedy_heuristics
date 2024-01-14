import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_encoder_layers, num_decoder_layers, output_dim, num_outputs):
        super(TransformerModel, self).__init__()
        self.num_outputs = num_outputs
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Initial deep layers
        self.linear = nn.Linear(input_dim, hidden_dim)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # Transformer Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

        # Output layer
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        batch_size = src.shape[1]
        src = self.linear(src)
        memory = self.transformer_encoder(src)

        output_indices = []
        sumLogProbOfActions = []
        mask = torch.zeros(self.output_dim, device=src.device)  # Initialize mask
        tgt = torch.zeros(1, batch_size, self.hidden_dim, device=src.device)  # Initialize target
        for _ in range(self.num_outputs):
            decoder_output = self.transformer_decoder(tgt, memory)

            output = self.out(decoder_output)
            output -= mask.unsqueeze(0)*1e9
            output = F.softmax(output, dim=2)
            output_idx = torch.argmax(output, dim=2)
            output_indices.append(output_idx)
            sumLogProbOfActions.append(torch.log(output[0, :, output_idx[0, :]])[0])
            mask[output_idx] = 1  # Update mask
            #tgt = last output
            tgt = decoder_output[-1].unsqueeze(0)



        return output_indices, torch.sum(torch.stack(sumLogProbOfActions))


# Model parameters
input_dim = 3
hidden_dim = 64
nhead = 4
num_encoder_layers = 1
num_decoder_layers = 1
output_dim = 200
num_outputs = 100
batch_size = 1

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coordinates = torch.rand(output_dim, 2).to(device)
    costs = torch.rand(output_dim).to(device)
    #rand input shape output_dim, batch_size, input_dim
    random_input = torch.zeros(output_dim, batch_size, input_dim).to(device)
    random_input[:, :, :2] = coordinates.unsqueeze(1)
    random_input[:, :, 2] = costs.unsqueeze(1)
    distance_matrix = torch.cdist(coordinates, coordinates, p=2)
    model = TransformerModel(input_dim, hidden_dim, nhead, num_encoder_layers, num_decoder_layers, output_dim,
                             num_outputs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for _ in range(1000):
        output_indices, sumLogProbOfActions = model(random_input)
        #to list
        output_indices = [output_indices[i].tolist()[0][0] for i in range(len(output_indices))]

        # Calculate cost
        distance_cost = torch.sum(distance_matrix[output_indices[:-1], output_indices[1:]]) + distance_matrix[output_indices[-1], output_indices[0]]
        total_cost = distance_cost + torch.sum(costs[output_indices])

        loss = -torch.mean(((total_cost-0)/num_outputs) * sumLogProbOfActions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Loss: {}".format(loss), )
        if _ % 10 == 0:
            plt.plot(coordinates[output_indices, 0].tolist(), coordinates[output_indices, 1].tolist())
            plt.title("Total cost: {}".format(total_cost))
            plt.show()


