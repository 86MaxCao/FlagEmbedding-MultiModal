import torch
from transformers import AutoModel
from PIL import Image

MODEL_NAME= "BAAI/BGE-VL-MLLM-S1"

model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.eval()
model.cuda()

with torch.no_grad():
    model.set_processor(MODEL_NAME)

    query_inputs = model.data_process(
        text="Make the background dark, as if the camera has taken the photo at night", 
        images="./assets/cir_query.png",
        q_or_c="q",
        task_instruction="Retrieve the target image that best meets the combined criteria by using both the provided image and the image retrieval instructions: "
    )

    candidate_inputs = model.data_process(
        images=["./assets/cir_candi_1.png", "./assets/cir_candi_2.png"],
        q_or_c="c",
    )

    query_embs = model(**query_inputs, output_hidden_states=True)[:, -1, :]
    candi_embs = model(**candidate_inputs, output_hidden_states=True)[:, -1, :]
    
    query_embs = torch.nn.functional.normalize(query_embs, dim=-1)
    candi_embs = torch.nn.functional.normalize(candi_embs, dim=-1)

    scores = torch.matmul(query_embs, candi_embs.T)
print(scores)
