import torch

@torch.no_grad()
def greedy_decode_seq2seq(
    model,
    src,
    sos_idx,
    eos_idx,
    max_len=50,
):
    """
    src: (1, T_src)
    returns: list of token ids
    """
    model.eval()
    device = src.device
    encoder_out, hidden = model.encoder(src)
    decoder_input = torch.tensor([[sos_idx]], device=device)

    generated = [sos_idx]

    for _ in range(max_len):
        logits, hidden, _ = model.decoder(
            decoder_input, hidden, encoder_out
        )

        next_token = logits.argmax(dim=-1).item()
        generated.append(next_token)

        if next_token == eos_idx:
            break

        decoder_input = torch.tensor([[next_token]], device=device)

    return generated

def translate_sentence_greedy_seq2seq(
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

    pred_ids = greedy_decode_seq2seq(
        model=model,
        src=src_tensor,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        max_len=max_len,
    )

    words = []
    for idx in pred_ids:
        tok = itos_tgt.get(idx, "<unk>")
        if tok == "<eos>":
            break
        if tok != "<sos>":
            words.append(tok)

    return " ".join(words)