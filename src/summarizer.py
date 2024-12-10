from transformers import BartForConditionalGeneration, BartTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text, max_length=200, min_length=100, num_beams=4):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs, 
        max_length=max_length, 
        min_length=min_length, 
        num_beams=num_beams
    )
    text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    text = text.replace("summarize:", "").strip()

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
