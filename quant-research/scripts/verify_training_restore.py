"""Verify a restored immutable package and materialize the hash-identical feature file."""
import argparse,json,os
from pathlib import Path
from quant_research.storage import file_hash,write_json
p=argparse.ArgumentParser();p.add_argument('restored',type=Path);p.add_argument('report',type=Path);a=p.parse_args()
m=json.loads((a.restored/'artifacts/package-inventory.json').read_text())
for name,sha in m['files'].items():
 assert file_hash(a.restored/name)==sha,name
r=m['materialize_duplicate'];source=a.restored/r['source'];target=a.restored/r['destination']
assert file_hash(source)==r['sha256']
if not target.exists():os.link(source,target)
assert file_hash(target)==r['sha256']
w=a.restored/'artifacts/full-market-training-20260910-v1'
for h in [5,20]:
 pm=json.loads((w/f'panel-h{h}/manifest.json').read_text())
 for name,sha in pm['files'].items():assert file_hash(w/f'panel-h{h}'/name)==sha
write_json(a.report,{'passed':True,'files_compared':len(m['files']),'both_panel_manifests_verified':True,'duplicate_feature_file_materialized':True})
print('restored inputs verified',len(m['files']))
