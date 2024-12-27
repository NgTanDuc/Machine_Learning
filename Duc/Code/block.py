import torch 
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        return self.conv_layers(x)
    
class downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super(downsample_block, self).__init__()
        self.downsample_layers = nn.Sequential(
            nn.MaxPool2d(2, padding=0),
            conv_block(in_channels, out_channels),
            conv_block(in_channels, out_channels),   
        )

        self.embedded_layers = nn.Sequential(
            nn.Linear(embedding_dim, out_channels),
            nn.SiLU()
        )

    def forward(self, x, embedding):
        x = self.downsample_layers(x)

        emb_block = self.embedded_layers(embedding)
        emb_block = emb_block.unsqueeze(-1).unsqueeze(-1)

        x = x + emb_block

        return x
    
class upsample_block(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, embedding_dim):
        super(upsample_block, self).__init__()
        
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.embedded_layers = nn.Sequential(
            nn.Linear(embedding_dim, out_channels),
            nn.SiLU()
        )

        self.input_skip = in_channels + skip_channels

        self.conv1 = conv_block(self.input_skip, out_channels)
        self.conv2 = conv_block(self.input_skip, out_channels)

    def forward(self, x, skip_connection, embedding):
        x = self.upsample_layer(x)

        x = torch.cat([x, skip_connection], dim=1)

        x = self.conv1(x)

        x = self.conv2(x)

        emb_block = self.embedded_layers(embedding)
        emb_block = emb_block.unsqueeze(-1).unsqueeze(-1)

        x = x + emb_block

        return x
    
class self_attention_block(nn.Module):
    def __init__(self):
        super(self_attention_block, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=[64])
        self.multi =  nn.MultiheadAttention(64, 4)
        
        self.block = nn.Sequential(
            nn.LayerNorm([64]),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

    def forward(self, x, in_size=64, channels=64):
        input = x.view(in_size*in_size, channels)

        x = self.layer_norm(input)
        x = self.multi(x)

        x1 = input + x

        x_block = self.block(x1) 

        output = x + x_block
        output = output.view(in_size, in_size, channels)

        return output