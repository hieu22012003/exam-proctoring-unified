"""Download the original Vietnamese Zipformer INT8 runtime with pinned checksums."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2'
SHA256='da8b637947091829d7ee9eda23da2a4ec7caa399233a3f4e34eb719fb2ea6b9b'
FILES={'encoder.int8.onnx','decoder.onnx','joiner.int8.onnx','tokens.txt','bpe.model','README.md'}


def main():
    dest=Path(os.getenv('EXAMWATCH_MODELS',str(ROOT/'models')))/'zipformer-vi-30m';dest.mkdir(parents=True,exist_ok=True)
    if all((dest/name).is_file() for name in FILES):
        print('Original Zipformer INT8 already prepared')
        return
    cache=ROOT/'.tmp';cache.mkdir(exist_ok=True)
    archive=cache/'zipformer-vi-30m.tar.bz2'
    if not archive.exists():
        urllib.request.urlretrieve(URL,archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Zipformer archive checksum mismatch; remove corrupt cached archive and retry')
    found=set()
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            name=Path(member.name).name
            if member.isfile() and name in FILES:
                (dest/name).write_bytes(source.extractfile(member).read());found.add(name)
    if found!=FILES:raise ValueError('Model archive is missing required files')
    (dest/'source.json').write_text(json.dumps(dict(url=URL,source_model='hynt/Zipformer-30M-RNNT-6000h',
        license='CC-BY-NC-ND-4.0',archive_sha256=SHA256,
        files={name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in FILES}),indent=2),encoding='utf-8')
    print('Original Zipformer INT8 ready in models/zipformer-vi-30m')


if __name__=='__main__':main()
