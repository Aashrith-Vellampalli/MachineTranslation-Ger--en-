import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self,hidden_dim,vocab_size,num_layers):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        
        self.embed=nn.Embedding(vocab_size,hidden_dim)
        self.lstm=nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self,x,hidden=None):
        x=self.embed(x)
        out,hidden=self.lstm(x,hidden)

        return out,hidden

class Attention(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.atten=nn.Linear(hidden_dim,hidden_dim,bias=False)

    def forward(self,encoder_out,decoder_hid):
        """
        encoder_out will be (B,T_src,hidden_dim)
        decoder will be (B,hidden_dim)

        atn_weight is (B,T_src)
        """
        encoder_proj=self.atten(encoder_out)
        similarity=torch.einsum("BTH,BH->BT",[encoder_proj,decoder_hid])

        atn_weights=torch.softmax(similarity,dim=1)
        context=torch.einsum("BT,BTH->BH",[atn_weights,encoder_out])

        return context,atn_weights

class Decoder(nn.Module):
    def __init__(self,hidden_dim,vocab_size,num_layers):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        self.num_layers=num_layers

        self.embed=nn.Embedding(vocab_size,hidden_dim)
        self.lstm=nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.attention=Attention(hidden_dim)
        self.fc=nn.Linear(hidden_dim*2,vocab_size)

    def forward(self,x,hidden,encoder_out):
        """
        out->(B,1,hidden_dim)
        """
        
        x=self.embed(x)
        out,hidden=self.lstm(x,hidden)
        
        context,atn_weights=self.attention(encoder_out,out.squeeze(1))
        concat=torch.cat((out.squeeze(1),context),dim=1)
        logits=self.fc(concat)

        return logits,hidden,atn_weights


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder,tf_ratio):
        super().__init__()

        self.encoder=encoder
        self.decoder=decoder
        self.tf_ratio=tf_ratio

    def forward(self,x,y):
        """
        x->(B,T_src)
        y->(B,T-tgt)
        """
        vocab_size=self.decoder.fc.out_features
        B,T_Y=y.shape
        outputs=torch.zeros(B,T_Y,vocab_size)

        encoder_out,hidden=self.encoder(x)
        decoder_input=y[:,0:1]

        for t in range(1,T_Y):
            logits,hidden,atn_weights=self.decoder(decoder_input,hidden,encoder_out)

            outputs[:,t]=logits
            teacher_force=random.random()<self.tf_ratio
            best=logits.argmax(dim=-1).unsqueeze(1)

            decoder_input= y[:,t].unsqueeze(1) if teacher_force else best

        return outputs





