"""Record or verify an immutable archive receipt after a visible Git capture."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('benchmark',type=Path);p.add_argument('archive',type=Path)
p.add_argument('url');p.add_argument('revision');a=p.parse_args()
receipt={'source_url':a.url,'revision':a.revision,'path':str(a.archive.relative_to(a.benchmark)),
         'size':a.archive.stat().st_size,'sha256':hashlib.file_digest(a.archive.open('rb'),'sha256').hexdigest(),
         'capture_method':'git archive of explicitly fetched revision'}
path=Path(str(a.archive)+'.receipt.json')
if path.exists():
 if json.loads(path.read_text())!=receipt:raise ValueError('Existing capture receipt differs; never overwrite raw inputs')
else:
 with path.open('x') as out:out.write(json.dumps(receipt,indent=2)+'\n')
