"""Download the original Vietnamese Zipformer INT8 runtime with pinned checksums."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2'
SHA256='da8b637947091829d7ee9eda23da2a4ec7caa399233a3f4e34eb719fb2ea6b9b'
FILES={'encoder.int8.onnx','decoder.onnx','joiner.int8.onnx','tokens.txt','bpe.model','README.md'}
# Same files, published by the sherpa-onnx author on Hugging Face. Used when the
# GitHub release asset is unreachable (it answered 504 to Render's builders).
MIRROR='https://huggingface.co/csukuangfj2/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/resolve/main/'
FILE_SHA256={  # from models/zipformer-vi-30m/source.json (extracted from the pinned archive)
    'tokens.txt':'ca8171f8bbd516c050b627582f2125c8f5f1f6ed967ab41b0fa9aae2cf61b492',
    'encoder.int8.onnx':'8ef5286dd427eb108055c2ddc1982aa31e544706072d5ea228729292dacade68',
    'README.md':'25ef401db87cfa8025c7241a4b5eaa53835b84358177e1e70a3c98393e83a0bd',
    'decoder.onnx':'cf2aa385b82c9d5d40cd29c3188af52d0249b3b78f0d4b7eb84ad502d50c7e7f',
    'bpe.model':'002894e7a82d80ffa5e25008ec8c5496159db804005e2103de96b01b4c13d445',
    'joiner.int8.onnx':'7311d2e17b810ecea515d79c71cc4668af8759256a06fa01d27047772320c821'}


def fetch(url,target,attempts=3):
    error=None
    for attempt in range(attempts):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'ExamWatch-Local/1.0'})
            partial=target.with_name(target.name+'.part')
            with urllib.request.urlopen(request,timeout=120) as source, open(partial,'wb') as out:
                while chunk:=source.read(1<<20):out.write(chunk)
            partial.replace(target)
            return
        except Exception as exc:
            error=exc
            wait=min(30,5*2**attempt)
            print(f'{target.name}: attempt {attempt+1} failed ({exc}); retrying in {wait}s',flush=True)
            time.sleep(wait)
    raise RuntimeError(f'Could not download {url}: {error}')


def from_mirror(dest):
    for name,expected in FILE_SHA256.items():
        fetch(MIRROR+name,dest/name)
        if hashlib.sha256((dest/name).read_bytes()).hexdigest()!=expected:
            if name=='README.md':  # documentation only; never blocks the build
                continue
            (dest/name).unlink()
            raise ValueError(f'Checksum mismatch for mirrored {name}')
    print('Zipformer INT8 downloaded from the Hugging Face mirror (checksums verified)',flush=True)


def main():
    dest=Path(os.getenv('EXAMWATCH_MODELS',str(ROOT/'models')))/'zipformer-vi-30m';dest.mkdir(parents=True,exist_ok=True)
    if all((dest/name).is_file() for name in FILES):
        print('Original Zipformer INT8 already prepared')
        return
    cache=ROOT/'.tmp';cache.mkdir(exist_ok=True)
    archive=cache/'zipformer-vi-30m.tar.bz2'
    source_url=URL
    try:
        if not archive.exists():
            fetch(URL,archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=SHA256:
            archive.unlink()
            raise ValueError('Zipformer archive checksum mismatch')
    except Exception as exc:
        print(f'GitHub archive unavailable ({exc}); using the Hugging Face mirror',flush=True)
        from_mirror(dest)
        source_url=MIRROR
    if source_url==URL:
        found=set()
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                name=Path(member.name).name
                if member.isfile() and name in FILES:
                    (dest/name).write_bytes(source.extractfile(member).read());found.add(name)
        if found!=FILES:raise ValueError('Model archive is missing required files')
    (dest/'source.json').write_text(json.dumps(dict(url=source_url,source_model='hynt/Zipformer-30M-RNNT-6000h',
        license='CC-BY-NC-ND-4.0',archive_sha256=SHA256,
        files={name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in FILES}),indent=2),encoding='utf-8')
    print('Original Zipformer INT8 ready in models/zipformer-vi-30m')


if __name__=='__main__':main()
