import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
from moreact.config import dump_json
from moreact.semantics.common import read_manifest, tensor_digest, sha256
from moreact.semantics.language import build_language, batch, texts_for_cache, audit_lengths, train_captioner
from moreact.semantics.diagnostics import reconstruction_report

root=Path('tmp/shared_rvq_validation')
torch.set_num_threads(4)
m=read_manifest(root/'real_tokens','tokens')
report=reconstruction_report(root/'real_latents/vae_snapshot.pt',root/'real_rvq/best.pt','data/interx_h2_f8',m,split='val',limit=1)
dump_json(root/'real_reconstruction.json',report)
print('Real FK reconstruction completed',flush=True)
# Load the actual local FLAN-T5-large weights; one no-grad forward only.
tokenizer,model=build_language('/data/autovla/projects/models/flan-t5-large',m['rvq_config'],m['identity']['future'],'cpu')
rows=texts_for_cache(root/'real_tokens',m)
audit_lengths(tokenizer,rows,root/'real_length_audit.json')
model.eval()
inputs,labels=batch(tokenizer,[rows[0]['text']],[rows[0]['captions'][0]],'cpu')
with torch.no_grad():
 loss=float(model(**inputs,labels=labels).loss)
assert torch.isfinite(torch.tensor(loss))
dump_json(root/'real_t5_forward.json',{'model':'local FLAN-T5-large','loss':loss,'input_tokens':inputs.input_ids.shape[1],'target_tokens':int((labels!=-100).sum()),'updated_parameters':False,'purpose':'asset and wiring check, not semantic quality'})
print('Real FLAN-T5-large forward completed',loss,flush=True)
del model,inputs,labels
# An explicitly random tiny T5 with the production tokenizer checks all CLI stages cheaply.
base=root/'tiny_language_base'
base.mkdir()
tokenizer=AutoTokenizer.from_pretrained('/data/autovla/projects/models/flan-t5-large',local_files_only=True,use_fast=True)
tokenizer.save_pretrained(str(base))
torch.manual_seed(9)
cfg=T5Config(vocab_size=len(tokenizer),d_model=16,d_ff=32,d_kv=8,num_layers=1,num_decoder_layers=1,num_heads=2,dropout_rate=0.,pad_token_id=0,eos_token_id=1,decoder_start_token_id=0)
T5ForConditionalGeneration(cfg).save_pretrained(str(base))
print('Random tiny language fixture created',flush=True)
