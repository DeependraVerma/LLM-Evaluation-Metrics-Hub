import os
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import nltk
from nltk.util import ngrams
import torch
from nltk.tokenize import word_tokenize
from collections import Counter
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure required packages are downloaded
nltk.download('punkt')

class LLM_Evaluator:
    def __init__(self):
        self.gpt2_model, self.gpt2_tokenizer = self.load_gpt2_model()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def load_gpt2_model(self):
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        return model, tokenizer

    def evaluate_bleu_rouge(self, candidates, references):
        bleu_score = corpus_bleu(candidates, [references]).score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
        rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
        return bleu_score, rouge1

    def evaluate_bert_score(self, candidates, references):
        P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
        return P.mean().item(), R.mean().item(), F1.mean().item()

    def evaluate_perplexity(self, text):
        encodings = self.gpt2_tokenizer(text, return_tensors='pt')
        max_length = self.gpt2_model.config.n_positions
        stride = 512
        lls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=target_ids)
                log_likelihood = outputs[0] * trg_len
            lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        return ppl.item()

    def evaluate_diversity(self, texts):
        all_tokens = [tok for text in texts for tok in text.split()]
        unique_bigrams = set(ngrams(all_tokens, 2))
        diversity_score = len(unique_bigrams) / len(all_tokens) if all_tokens else 0
        return diversity_score
    
    def calculate_burstiness(self, text):
        tokens = word_tokenize(text.lower())
        token_counts = Counter(tokens)
        max_freq = max(token_counts.values())
        burstiness_score = max_freq / len(tokens)
        return burstiness_score
    
    def evaluate_readability(self, text):
        return flesch_reading_ease(text)
    
    def evaluate_sentiment(self, text):
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        return sentiment['compound']

    def evaluate_all(self, response, reference):
        candidates = [response]
        references = [reference]
        bleu, rouge1 = self.evaluate_bleu_rouge(candidates, references)
        bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
        perplexity = self.evaluate_perplexity(response)
        diversity = self.evaluate_diversity(candidates)
        burstiness_score = self.calculate_burstiness(response)
        readability = self.evaluate_readability(response)
        sentiment = self.evaluate_sentiment(response)
        
        return {
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "BERT P": bert_p,
            "BERT R": bert_r,
            "BERT F1": bert_f1,
            "Perplexity": perplexity,
            "Diversity": diversity,
            "Burstiness": burstiness_score,
            "Readability": readability,
            "Sentiment": sentiment,
        }
