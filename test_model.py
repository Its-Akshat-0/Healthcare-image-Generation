
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_PATH = "models/ruralhealth-qa"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
model.eval()

context = """
Dengue is a viral infection transmitted to humans through
the bites of infected Aedes mosquitoes. Prevention includes
avoiding mosquito bites, using mosquito repellent, wearing
protective clothing, and reducing standing water where
mosquitoes can breed.
"""

question = "How is dengue transmitted?"

inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

with torch.no_grad():
    outputs = model(**inputs)

start_index = torch.argmax(outputs.start_logits, dim=1).item()
end_index = torch.argmax(outputs.end_logits, dim=1).item()

if end_index < start_index:
    print("The model could not extract a valid answer.")
else:
    answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    print("Question:", question)
    print("Answer:", answer)
    print("Start score:", round(outputs.start_logits[0, start_index].item(), 4))
    print("End score:", round(outputs.end_logits[0, end_index].item(), 4))
