import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super().__init__()

        self.embed_size=embed_size
        self.heads=heads
        self.head_dim=embed_size//heads

        self.W_q=nn.Linear(embed_size,embed_size,bias=False)
        self.W_k=nn.Linear(embed_size,embed_size,bias=False)
        self.W_v=nn.Linear(embed_size,embed_size,bias=False)
        
        self.fc=nn.Linear(embed_size,embed_size)

    def forward(self,values,keys,queries,mask):
        B,T_q,_=queries.shape
        B,T_k,_=keys.shape

        Q=self.W_q(queries)
        K=self.W_k(keys)
        V=self.W_v(values)

        Q=Q.view(B,T_q,self.heads,self.head_dim).transpose(1,2)
        K=K.view(B,T_k,self.heads,self.head_dim).transpose(1,2)
        V=V.view(B,T_k,self.heads,self.head_dim).transpose(1,2)

        energy=torch.einsum("bhqd,bhkd->bhqk",[Q,K])
        energy=energy/(self.head_dim**0.5)

        if mask is not None:
            energy=energy.masked_fill(mask==0,-1e20)

        attention=torch.softmax(energy,dim=-1)
        out=torch.einsum("bhqk,bhkd->bhqd",[attention,V])
        out=out.transpose(1,2).contiguous().view(B,T_q,self.embed_size)
        out=self.fc(out)

        return out
    
class EncoderBlock(nn.Module):
    def __init__(self,embed_size,heads,expansion=4,dropout=0.1):
        super().__init__()
        self.embed_size=embed_size
        self.heads=heads
        
        self.attention=MultiHeadAttention(embed_size,heads)
        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)

        self.ffn=nn.Sequential(
            nn.Linear(embed_size,embed_size*expansion),
            nn.ReLU(),
            nn.Linear(embed_size*expansion,embed_size)
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        out=self.attention(x,x,x,mask)
        out=self.norm1(x+self.dropout(out))
        
        out2=self.ffn(out)
        out2=self.norm2(self.dropout(out2)+out)

        return out2



class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_size,max_len,num_layers=2,heads=8,expansion=4,dropout=0.1):
        super().__init__()
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.heads=heads
        
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.position_embed=nn.Embedding(max_len,embed_size)
        self.layers=nn.ModuleList(
            [
                EncoderBlock(embed_size,heads,expansion,dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,mask=None):
        """ 
        x->(B,T)
        mask should be (B,1,1,T) or None
        """
        B,T=x.shape
        positions=torch.arange(0,T).unsqueeze(0).expand(B,T)
        out=self.embed(x)*(self.embed_size ** 0.5)+self.position_embed(positions)
        out=self.dropout(out)

        for layer in self.layers:
            out=layer(out,mask)
        
        return out



class DecoderBlock(nn.Module):
    def __init__(self,embed_size,heads,expansion=4,dropout=0.1):
        super().__init__()
        self.heads=heads
        self.embed_size=embed_size

        self.self_atn=MultiHeadAttention(embed_size,heads)
        self.cross_atn=MultiHeadAttention(embed_size,heads)

        self.norm1=nn.LayerNorm(embed_size)
        self.norm2=nn.LayerNorm(embed_size)
        self.norm3=nn.LayerNorm(embed_size)

        self.ffn=nn.Sequential(
            nn.Linear(embed_size,embed_size*expansion),
            nn.ReLU(),
            nn.Linear(embed_size*expansion,embed_size),
        )
        self.dropout=nn.Dropout(dropout)


    def forward(self,x,encoder_out,src_mask,tgt_mask):
        out1=self.self_atn(x,x,x,tgt_mask)
        x=self.norm1(x+self.dropout(out1))
        
        out2=self.cross_atn(encoder_out,encoder_out,x,src_mask)
        x=self.norm2(x+self.dropout(out2))

        out3=self.ffn(x)
        x=self.norm3(x+self.dropout(out3))

        return x
    

class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,max_len,num_layers=2,heads=8,expansion=4,dropout=0.1):
        super().__init__()
        self.vocab_size=vocab_size
        self.embed_size=embed_size

        self.pos_embed=nn.Embedding(max_len,embed_size)
        self.embed=nn.Embedding(vocab_size,embed_size)

        self.layers=nn.ModuleList(
            DecoderBlock(embed_size,heads,expansion,dropout)
            for _ in range(num_layers)
        )

        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(embed_size,vocab_size)
        self.fc.weight=self.embed.weight

    def forward(self,x,encoder_out,src_mask,tgt_mask):
        """
        x->(B,T_tgt)
        encoder_out->(B,T_src,embed_size)
        src_mask->(B,1,1,T_src)
        tgt_mask->(B,1,T_tgt,T_tgt)
        """

        B,T=x.shape
        positions=torch.arange(0,T).unsqueeze(0).expand(B,T)
        x=self.embed(x)*(self.embed_size**0.5)+self.pos_embed(positions)
        x=self.dropout(x)

        for layer in self.layers:
            x=layer(x,encoder_out,src_mask,tgt_mask)
        
        logits=self.fc(x)

        return logits

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,embed_size,heads,num_layers,
                 expansion=4,dropout=0.1,max_len=50,pad_idx=0):
        super().__init__()

        self.pad_idx=pad_idx

        self.encoder=Encoder(
            vocab_size=src_vocab_size,
            embed_size=embed_size,
            max_len=max_len,
            heads=heads,
            num_layers=num_layers,
            expansion=expansion,
            dropout=dropout
        )

        self.decoder=Decoder(
            vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            max_len=max_len,
            heads=heads,
            num_layers=num_layers,
            expansion=expansion,
            dropout=dropout
        )

    def make_src_mask(self,src):
        src_mask=(src!=self.pad_idx)
        return src_mask.unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        B,T=tgt.shape

        causal_mask=torch.tril(torch.ones((T, T), device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) 

        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        return causal_mask & pad_mask & pad_mask.transpose(-1, -2)
    
    def forward(self,src,tgt):
        src_mask=self.make_src_mask(src)
        tgt_mask=self.make_tgt_mask(tgt)

        encoder_out=self.encoder(src,src_mask)
        out=self.decoder(tgt,encoder_out,src_mask,tgt_mask)

        return out


class NoamLR:
    def __init__(self, optimizer, model_dim, warmup_steps, step_num=0):
        self.optimizer = optimizer
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
        self.step_num = step_num

    def step(self):
        self.step_num += 1
        lr = (
            self.model_dim ** (-0.5)
            * min(
                self.step_num ** (-0.5),
                self.step_num * (self.warmup_steps ** -1.5),
            )
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr