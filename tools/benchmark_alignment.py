#!/usr/bin/env python3
"""Benchmark identical CTC inputs in isolated CPU workers; report time, RSS and timestamps.

Example: python tools/benchmark_alignment.py --audio jfk.flac --text 'and so ...' \
  --onnx-fp32 /tmp/float-bundle --onnx-int8 /tmp/int8-bundle --output evidence.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

# Allow invocation directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compare_words(reference, candidate):
    """Require every word and finite boundaries before reporting drift."""
    import math
    if [w['word'] for w in reference] != [w['word'] for w in candidate]:
        return {'comparable': False, 'reason': 'word_sequence_changed'}
    errors = []
    for left, right in zip(reference, candidate):
        for key in ('start', 'end'):
            if key not in left or key not in right or not math.isfinite(left[key]) or not math.isfinite(right[key]):
                return {'comparable': False, 'reason': 'missing_or_nonfinite_timestamp'}
            errors.append(abs(left[key] - right[key]) * 1000)
    if not errors:
        return {'comparable': False, 'reason': 'no_words'}
    errors.sort()
    return {'comparable': True, 'boundary_mean_ms': statistics.mean(errors),
            'boundary_p95_ms': errors[min(len(errors)-1, int(len(errors)*.95))],
            'boundary_max_ms': max(errors)}


def worker(args):
    import resource
    import numpy as np
    import torch
    from whisperx.alignment import align, load_align_model
    from whisperx.alignment_engine import get_emission_engine
    from whisperx.audio import load_audio, SAMPLE_RATE

    torch.set_num_threads(args.threads)
    audio = load_audio(args.audio)
    text = args.text
    if args.duration:
        copies = max(1, int(args.duration * SAMPLE_RATE) // len(audio))
        audio = np.tile(audio, copies)
        text = ' '.join([text] * copies)
        target = int(args.duration * SAMPLE_RATE)
        audio = np.pad(audio, (0, max(0, target-len(audio))))
    started = time.perf_counter()
    model, metadata = load_align_model(args.language, 'cpu', model_name=args.model,
                                      quantize=args.variant == 'torch-int8',
                                      model_cache_only=True,
                                      engine='onnx' if args.variant.startswith('onnx') else 'torch',
                                      onnx_threads=args.threads)
    load_ms = (time.perf_counter()-started)*1000
    if args.variant == 'torch-int8' and not any(
            isinstance(module, torch.ao.nn.quantized.dynamic.Linear) for module in model.modules()):
        raise RuntimeError('Requested Torch int8 but no quantized Linear layers were loaded')
    if args.variant.startswith('onnx') and bool(metadata.get('quantized')) != (args.variant == 'onnx-int8'):
        raise RuntimeError('ONNX bundle quantization does not match the requested benchmark variant')
    backend = get_emission_engine(model, metadata)

    class TimedEngine:
        stride = backend.stride
        receptive_field = backend.receptive_field
        forward_ms = 0
        calls = 0

        def infer(self, waveforms, lengths):
            started = time.perf_counter()
            result = backend.infer(waveforms, lengths)
            self.forward_ms += (time.perf_counter()-started)*1000
            self.calls += 1
            return result

    engine = TimedEngine()
    transcript = [{'start': 0, 'end': len(audio)/SAMPLE_RATE, 'text': text}]
    times, forwards, rust_times = [], [], []
    import whisperx.alignment as alignment
    original_rust = alignment._rust_align_segment
    rust_ms = 0

    def timed_rust(*a, **kw):
        nonlocal rust_ms
        started = time.perf_counter()
        try:
            return original_rust(*a, **kw)
        finally:
            rust_ms += (time.perf_counter()-started)*1000

    alignment._rust_align_segment = timed_rust
    try:
        for repeat in range(args.warmup + args.repeats):
            engine.forward_ms = 0
            engine.calls = 0
            rust_ms = 0
            started = time.perf_counter()
            result = align(transcript, engine, {**metadata, 'type': 'custom'}, audio, 'cpu',
                           batch_size=1, chunk_seconds=args.chunk or None,
                           return_diagnostics=True)
            elapsed = (time.perf_counter()-started)*1000
            if repeat >= args.warmup:
                times.append(elapsed)
                forwards.append(engine.forward_ms)
                rust_times.append(rust_ms)
    finally:
        alignment._rust_align_segment = original_rust
    for word in result['word_segments']:
        if not (0 <= word.get('start', -1) <= word.get('end', -1) <= len(audio)/SAMPLE_RATE + .001):
            raise RuntimeError(f"Invalid final word timestamps: {word}")
    cgroup = {}
    for name in ('memory.max', 'memory.peak', 'memory.events', 'cpu.max'):
        path = Path('/sys/fs/cgroup')/name
        if path.exists():
            cgroup[name] = path.read_text().strip()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == 'darwin' else peak*1024
    record = {'variant': args.variant, 'chunk_seconds': args.chunk or None,
              'audio_seconds': len(audio)/SAMPLE_RATE, 'load_ms': load_ms,
              'align_ms': times, 'align_median_ms': statistics.median(times),
              'forward_median_ms': statistics.median(forwards), 'rust_median_ms': statistics.median(rust_times),
              'overhead_median_ms': statistics.median([t-f-r for t,f,r in zip(times, forwards, rust_times)]),
              'peak_rss_bytes': peak_bytes, 'forward_calls': engine.calls,
              'words': result['word_segments'], 'diagnostics': result['diagnostics'],
              'torch': torch.__version__, 'python': platform.python_version(),
              'platform': platform.platform(), 'processor': platform.processor(),
              'threads': args.threads, 'quantized_engine': torch.backends.quantized.engine,
              'audio_sha256': hashlib.sha256(Path(args.audio).read_bytes()).hexdigest(),
              'text_sha256': hashlib.sha256(args.text.encode()).hexdigest(),
              'cgroup': cgroup, 'model_name': metadata.get('model_name', args.model)}
    if args.variant.startswith('onnx'):
        import onnxruntime
        record['onnxruntime'] = onnxruntime.__version__
        record['bundle'] = metadata
    Path(args.output).write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--text', required=True, help='Exact spoken transcript')
    parser.add_argument('--language', default='en')
    parser.add_argument('--model')
    parser.add_argument('--onnx-fp32')
    parser.add_argument('--onnx-int8')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--durations', default='0,30,90', help='0 keeps original audio; longer cases repeat complete utterances and pad silence')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--max-boundary-ms', type=float, default=100)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check', action='store_true', help='Exit nonzero on worker failure or timestamp/memory budget violation')
    parser.add_argument('--max-rss-mib', type=float, default=0, help='Optional worker process peak RSS budget (0 disables)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--variant', default='torch-fp32', help=argparse.SUPPRESS)
    parser.add_argument('--chunk', type=float, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--duration', type=float, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.threads < 1:
        parser.error('repeats/threads must be positive and warmup nonnegative')
    if args.worker:
        worker(args)
        return
    variants = [('torch-fp32', args.model, 0), ('torch-int8', args.model, 0), ('torch-fp32', args.model, 10)]
    for name, path in [('onnx-fp32', args.onnx_fp32), ('onnx-int8', args.onnx_int8)]:
        if path:
            variants.extend([(name, path, 0), (name, path, 10)])
    records = []
    with tempfile.TemporaryDirectory(prefix='whisperx-bench-') as tmp:
        for duration in map(float, args.durations.split(',')):
            baseline = None
            for variant, model, chunk in variants:
                output = Path(tmp)/'worker.json'
                output.unlink(missing_ok=True)
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--audio', args.audio,
                           '--text', args.text, '--language', args.language, '--threads', str(args.threads),
                           '--repeats', str(args.repeats), '--warmup', str(args.warmup), '--variant', variant,
                           '--chunk', str(chunk), '--duration', str(duration), '--output', str(output)]
                if model:
                    command.extend(['--model', model])
                try:
                    process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout,
                                             env={**os.environ, 'HF_HUB_OFFLINE': '1'})
                    if process.returncode:
                        record = {'variant': variant, 'chunk_seconds': chunk or None, 'requested_seconds': duration,
                                  'error': 'worker_failed', 'returncode': process.returncode, 'stderr': process.stderr[-3000:]}
                    else:
                        record = json.loads(output.read_text())
                        if baseline is None and variant == 'torch-fp32' and not chunk:
                            baseline = record
                        if baseline:
                            record['fidelity'] = compare_words(baseline['words'], record['words'])
                            record['fidelity']['within_budget'] = (record['fidelity']['comparable'] and
                                record['fidelity']['boundary_max_ms'] <= args.max_boundary_ms)
                            record['speedup'] = baseline['align_median_ms']/record['align_median_ms']
                except subprocess.TimeoutExpired:
                    record = {'variant': variant, 'chunk_seconds': chunk or None, 'requested_seconds': duration, 'error': 'timeout'}
                if args.max_rss_mib and 'peak_rss_bytes' in record:
                    record['within_memory_budget'] = record['peak_rss_bytes'] <= args.max_rss_mib*2**20
                records.append(record)
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps({'schema_version': 1, 'warmup': args.warmup,
                    'max_boundary_ms': args.max_boundary_ms, 'records': records}, indent=2, allow_nan=False)+'\n')
                print(json.dumps({k:v for k,v in record.items() if k not in {'words','diagnostics','bundle','stderr'}}, sort_keys=True), flush=True)

    if args.check and any('error' in r or not r.get('fidelity', {}).get('within_budget', False) or
                          not r.get('within_memory_budget', True) for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
