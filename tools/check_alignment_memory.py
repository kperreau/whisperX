#!/usr/bin/env python3
"""Compare chunked/full ONNX inference in offline, memory-limited Docker workers.

The image must contain Torch, ONNX Runtime, FFmpeg and whisperx_ext. This checkout's
Python package and benchmark script are mounted read-only. Timings under emulation
are functional evidence, not native CPU performance measurements.
"""
import argparse
import json
from pathlib import Path
import subprocess
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--language', default='en')
    parser.add_argument('--platform', default='linux/amd64')
    parser.add_argument('--python', default='/opt/venv/bin/python')
    parser.add_argument('--memory', default='2g')
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--duration', type=float, default=90)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--reference-json', type=Path, help='Full native benchmark output to gate final word boundaries')
    parser.add_argument('--max-boundary-ms', type=float, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.bundle/'metadata.json').read_text())
    variant = 'onnx-int8' if metadata['quantized'] else 'onnx-fp32'
    records = []
    for label, chunk in [('chunked', 10), ('full', 0)]:
        command = ['docker', 'create', '--name', 'whisperx-memory-'+uuid.uuid4().hex[:10],
                   '--platform', args.platform, '--network', 'none', '--memory', args.memory,
                   '--memory-swap', args.memory, '--cpus', str(args.cpus),
                   '--entrypoint', args.python, '-e', 'PYTHONPATH=/checkout',
                   '-v', f'{root}/whisperx:/checkout/whisperx:ro',
                   '-v', f'{root}/tools:/checkout/tools:ro',
                   '-v', f'{args.audio.resolve()}:/fixture:ro',
                   '-v', f'{args.bundle.resolve()}:/bundle:ro',
                   '-v', f'{args.output_dir.resolve()}:/evidence', args.image,
                   '/checkout/tools/benchmark_alignment.py', '--worker', '--audio', '/fixture',
                   '--text', args.text, '--language', args.language, '--variant', variant,
                   '--model', '/bundle', '--chunk', str(chunk), '--duration', str(args.duration),
                   '--threads', str(args.cpus), '--warmup', '0', '--repeats', '1',
                   '--output', f'/evidence/{label}.json']
        container = subprocess.check_output(command, text=True).strip()
        timed_out = False
        stderr = ''
        try:
            try:
                process = subprocess.run(['docker', 'start', '-a', container], capture_output=True,
                                         text=True, timeout=args.timeout)
                stderr = process.stderr[-3000:]
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(['docker', 'kill', container], capture_output=True, check=True)
            info = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
            record = {'label': label, 'image': args.image, 'image_id': info['Image'],
                      'platform': args.platform, 'memory_limit': info['HostConfig']['Memory'],
                      'memory_swap_limit': info['HostConfig']['MemorySwap'],
                      'nano_cpus': info['HostConfig']['NanoCpus'],
                      'network_mode': info['HostConfig']['NetworkMode'],
                      'exit_code': info['State']['ExitCode'], 'oom_killed': info['State']['OOMKilled'],
                      'timed_out': timed_out, 'stderr': stderr}
            records.append(record)
            if label == 'chunked' and record['exit_code'] == 0 and args.reference_json:
                from benchmark_alignment import compare_words

                worker = json.loads((args.output_dir/'chunked.json').read_text())
                references = json.loads(args.reference_json.read_text())['records']
                reference = next(r for r in references if r.get('audio_seconds') == args.duration
                                 and r['variant'] == 'torch-fp32' and r['chunk_seconds'] is None)
                if (reference['audio_sha256'] != worker['audio_sha256'] or
                        reference['text_sha256'] != worker['text_sha256']):
                    raise ValueError('Reference audio/transcript hashes do not match the worker')
                record['fidelity'] = compare_words(reference['words'], worker['words'])
                record['fidelity']['within_budget'] = (
                    record['fidelity']['comparable'] and
                    record['fidelity']['boundary_max_ms'] <= args.max_boundary_ms)
            (args.output_dir/'container-states.json').write_text(json.dumps(records, indent=2)+'\n')
            print(json.dumps(record), flush=True)
        finally:
            subprocess.run(['docker', 'rm', '-f', container], capture_output=True, check=True)
    if (records[0]['exit_code'] or records[0]['oom_killed'] or records[0]['timed_out'] or
            not records[0].get('fidelity', {}).get('within_budget', True)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
