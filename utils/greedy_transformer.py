import torch
def ids_to_tokens(ids, itos):
    words = []
    for idx in ids:
        tok = itos.get(idx, "<unk>")
        if tok == "<eos>":
            break
        if tok != "<sos>":
            words.append(tok)
    return words



@torch.no_grad()
def greedy_decode_transformer(
    model,
    src,
    sos_idx,
    eos_idx,
    max_len=50,
):
    """
    src: (1, T_src)  tokenized + indexed source sentence
    """
    model.eval()
    device = src.device

    src_mask = model.make_src_mask(src)  
    encoder_out = model.encoder(src, src_mask)        

    tgt = torch.tensor([[sos_idx]], device=device)

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)        

        output = model.decoder(
            tgt, encoder_out, src_mask, tgt_mask
        )                                             

        logits = output[:, -1, :]                    
        next_token = logits.argmax(dim=-1)            

        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

        if next_token.item() == eos_idx:
            break

    return tgt.squeeze(0).tolist()


def translate_sentence_greedy_transformer(
    sentence,
    model,
    tokenize_src,
    vocab_src,
    itos_tgt,
    sos_idx,
    eos_idx,
    max_len=50,
):
    tokens = tokenize_src(sentence)
    src_ids = [vocab_src.get(tok, 1) for tok in tokens]
    src_ids = [sos_idx] + src_ids + [eos_idx]

    src_tensor = torch.tensor(src_ids).unsqueeze(0)

    pred_ids = greedy_decode_transformer(
        model=model,
        src=src_tensor,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        max_len=max_len,
    )

    return " ".join(ids_to_tokens(pred_ids, itos_tgt))