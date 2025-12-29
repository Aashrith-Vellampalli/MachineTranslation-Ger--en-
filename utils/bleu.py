from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

def evaluate_bleu(model,data,tokenize_src,tokenize_tgt,vocab_src,itos_tgt,sos_idx,
                  eos_idx,translate_fn):
    references=[]
    hypotheses=[]

    smooth=SmoothingFunction().method4

    for _, example in enumerate(data):
        src_sentence = example["de"]
        tgt_sentence = example["en"]

        pred_sentence = translate_fn(src_sentence,model,tokenize_src,vocab_src,itos_tgt,
                                    sos_idx,eos_idx)
        pred_tokens=pred_sentence.split()

        tgt_tokens = tokenize_tgt(tgt_sentence)

        references.append([tgt_tokens])
        hypotheses.append(pred_tokens)

    return corpus_bleu(references,hypotheses,smoothing_function=smooth)