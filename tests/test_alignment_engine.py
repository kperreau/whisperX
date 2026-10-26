"""Frame geometry, bounded inference, offline bundles and timestamp fidelity."""
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from whisperx.alignment_engine import TorchEmissionEngine, get_emission_engine, infer_emissions
from whisperx.alignment import align, load_align_model
from whisperx.alignment_text import prepare_alignment_text


class LocalEngine:
    stride = 320
    receptive_field = 400

    def __init__(self):
        self.calls = []

    def infer(self, waveform, lengths):
        self.calls.append((waveform.shape, lengths.tolist()))
        frames = (waveform.shape[1]-400)//320+1
        values = waveform[:, :frames*320:320]
        return torch.stack([values, -values, values*0], -1), (lengths-400)//320+1


@pytest.mark.parametrize('length', [1, 399, 400, 719, 720, 16000, 160003, 1440003])
def test_chunks_preserve_every_global_frame_once(length):
    waveform = torch.arange(length, dtype=torch.float32)/length
    engine = LocalEngine()
    full = infer_emissions([waveform], engine, 'cpu', chunk_seconds=None)[0]
    engine.calls.clear()
    chunks = infer_emissions([waveform], engine, 'cpu', chunk_seconds=2, context_seconds=.2)[0]
    np.testing.assert_array_equal(chunks, full)
    assert chunks.shape[0] == (max(length, 400)-400)//320+1
    assert max(shape[1] for shape, _ in engine.calls) <= 2.4*16000+400


def test_padding_budget_and_original_order():
    waveforms = [torch.full((length,), float(i)) for i, length in enumerate([64000, 400, 16000, 8000, 32000])]
    engine = LocalEngine()
    results = infer_emissions(waveforms, engine, 'cpu', batch_size=8, max_batch_seconds=3)
    for waveform, result in zip(waveforms, results):
        expected = infer_emissions([waveform], LocalEngine(), 'cpu')[0]
        np.testing.assert_array_equal(result, expected)
    assert all(shape[0] == 1 or shape[0]*shape[1] <= 3*16000 for shape, _ in engine.calls)


def test_empty_audio_and_invalid_chunk_configuration():
    for kwargs in ({'batch_size': 0}, {'chunk_seconds': 0}, {'context_seconds': -1}, {'max_batch_seconds': 0}):
        with pytest.raises(ValueError):
            infer_emissions([torch.zeros(1000)], LocalEngine(), 'cpu', **kwargs)
    with pytest.raises(ValueError, match='empty'):
        infer_emissions([torch.empty(0)], LocalEngine(), 'cpu')


def test_hf_exact_lengths_and_attention_mask():
    model = mock.Mock()
    model.return_value = SimpleNamespace(logits=torch.zeros(2, 49, 3))
    model._get_feat_extract_output_lengths.return_value = torch.tensor([49, 1])
    _, lengths = TorchEmissionEngine(model, 'huggingface').infer(torch.zeros(2, 16000), torch.tensor([16000, 400]))
    assert lengths.tolist() == [49, 1]
    assert model.call_args.kwargs['attention_mask'].sum(1).tolist() == [16000, 400]


@pytest.mark.parametrize('adapter', [False, True])
def test_hf_custom_convolution_grid_preserves_chunk_frames(adapter):
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    config = Wav2Vec2Config(
        hidden_size=8, output_hidden_size=8, num_hidden_layers=0, num_attention_heads=2,
        intermediate_size=16, conv_dim=(4, 4), conv_kernel=(5, 3), conv_stride=(4, 2),
        feat_extract_norm='layer', num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
        vocab_size=4, mask_time_prob=0, add_adapter=adapter, num_adapter_layers=2,
    )
    torch.manual_seed(21)
    model = Wav2Vec2ForCTC(config).eval()
    engine = get_emission_engine(model, {'type': 'huggingface'})
    assert engine.receptive_field == 13
    assert engine.stride == (32 if adapter else 8)
    waveform = torch.randn(5003)
    full = infer_emissions([waveform], engine, 'cpu', chunk_seconds=None)[0]
    chunked = infer_emissions(iter([waveform]), engine, 'cpu', chunk_seconds=.05, context_seconds=.01)[0]
    assert len(chunked) == int(model._get_feat_extract_output_lengths(torch.tensor([len(waveform)]))[0])
    np.testing.assert_allclose(chunked, full, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('logits, lengths', [
    (torch.zeros(1, 1, 2), [0]), (torch.zeros(1, 1, 2), [-1]),
    (torch.zeros(1, 1, 2), [2]), (torch.zeros(1, 1, 2), [1.5]),
    (torch.zeros(1, 1, 2), [1, 1]), (torch.zeros(2, 1, 2), [1, 1]),
    (torch.zeros(1, 2), None), (torch.full((1, 1, 2), float('nan')), [1]),
])
def test_invalid_engine_outputs_are_rejected(logits, lengths):
    engine = LocalEngine()
    engine.infer = lambda *_: (logits, lengths)
    with pytest.raises(ValueError, match='Emission'):
        infer_emissions([torch.zeros(400)], engine, 'cpu')


def test_torchaudio_cache_only_never_calls_downloader(tmp_path):
    bundle = mock.Mock(_path='missing.pt')
    torchaudio = SimpleNamespace(pipelines=SimpleNamespace(__all__=['fixture'], fixture=bundle))
    with mock.patch('whisperx.alignment._load_torchaudio', return_value=torchaudio):
        with pytest.raises(FileNotFoundError, match='not cached'):
            load_align_model('en', 'cpu', model_name='fixture', model_dir=tmp_path, model_cache_only=True)
    bundle.get_model.assert_not_called()


def test_replacements_preserve_display_offsets_and_diagnostics():
    chars, origins, diagnostics = prepare_alignment_text('cost 43 euros', dict.fromkeys('abcdefghijklmnopqrstuvwxyz|', 1),
        replacements=[{'start': 5, 'end': 7, 'text': 'forty three'}])
    assert ''.join(chars) == 'cost|forty|three|euros'
    assert origins[5:16] == [5]*11
    assert diagnostics['unsupported_characters'] == []
    assert diagnostics['words'][1]['method'] == 'spoken'


@pytest.mark.parametrize('replacement', [
    {'start': -1, 'end': 2, 'text': 'one'}, {'start': 1, 'end': 1, 'text': 'one'},
    {'start': 0, 'end': 4, 'text': 'one'}, {'start': 0, 'end': 1, 'text': ''},
])
def test_invalid_replacements_are_rejected(replacement):
    with pytest.raises(ValueError):
        prepare_alignment_text('1 2', {}, replacements=[replacement])


def test_spoken_replacement_matches_actual_word_boundaries():
    # Fixed, sharply peaked emissions represent 'a forty three b'. The displayed
    # number must span BOTH spoken words, not merely receive some timestamp.
    spoken = 'a forty three b'
    dictionary = {c: i+1 for i,c in enumerate(sorted(set(spoken.replace(' ', '|'))))}
    dictionary['<pad>'] = 0
    logits = torch.full((len(spoken)*4+8, len(dictionary)), -8.)
    logits[:, 0] = 2
    for i,c in enumerate(spoken.replace(' ', '|')):
        logits[4+i*4:6+i*4, dictionary[c]] = 8
        logits[4+i*4:6+i*4, 0] = -8
    model = mock.Mock(return_value=(logits[None], None))
    metadata = {'language': 'en', 'type': 'torchaudio', 'dictionary': dictionary}
    audio = np.zeros(16000*4, np.float32)
    reference = align([{'start': 0, 'end': 4, 'text': spoken}], model, metadata, audio, 'cpu')
    result = align([{'start': 0, 'end': 4, 'text': 'a 43 b', 'spoken_replacements': [
        {'start': 2, 'end': 4, 'text': 'forty three'}]}], model, metadata, audio, 'cpu', return_diagnostics=True)
    words = result['word_segments']
    assert [w['word'] for w in words] == ['a', '43', 'b']
    assert words[1]['start'] == reference['word_segments'][1]['start']
    assert words[1]['end'] == reference['word_segments'][2]['end']
    assert words[0] == reference['word_segments'][0]
    assert words[2] == reference['word_segments'][3]
    assert result['diagnostics'][0]['words'][1]['method'] == 'spoken'
    assert result['diagnostics'][0]['unsupported_characters'] == []


def test_unknown_policy_and_fallback_diagnostics():
    metadata = {'language': 'en', 'type': 'torchaudio', 'dictionary': {'a': 1, '<pad>': 0}}
    model = mock.Mock(return_value=(torch.zeros(1, 1, 2), None))
    with pytest.raises(ValueError, match='Unsupported alignment characters'):
        align([{'start': 0, 'end': 1, 'text': '43'}], model, metadata, np.zeros(16000, np.float32), 'cpu', unsupported_policy='error')
    model.assert_not_called()
    result = align(iter([{'start': 0, 'end': 1, 'text': 'aaa'}]), model, metadata,
                   np.zeros(16000, np.float32), 'cpu', return_diagnostics=True)
    assert result['diagnostics'][0]['reason'] == 'backtrack_failed'
    assert result['diagnostics'][0]['status'] == 'fallback'


def test_benchmark_rejects_changed_words_and_missing_boundaries():
    from tools.benchmark_alignment import compare_words

    reference = [{'word': 'forty', 'start': 1., 'end': 2.}]
    assert not compare_words(reference, [{'word': 'four', 'start': 1., 'end': 2.}])['comparable']
    assert not compare_words(reference, [{'word': 'forty', 'start': 1.}])['comparable']
    drift = compare_words(reference, [{'word': 'forty', 'start': 1.02, 'end': 2.1}])
    assert drift['boundary_max_ms'] == pytest.approx(100)
    assert drift['boundary_mean_ms'] == pytest.approx(60)


def test_profiler_does_not_modify_model_class_and_counts_total_once(capsys):
    from whisperx.profile_align import Profile, ProfiledAlign, _Bucket

    class Model:
        def __call__(self, waveform, lengths):
            return torch.tensor([[[0., 5.], [5., 0.]]]), None

    original = Model.__call__
    profiler = ProfiledAlign(np.zeros(16000, np.float32), Model(),
        {'type': 'torchaudio', 'language': 'en', 'dictionary': {'<pad>': 0, 'a': 1}})
    profiler.run([{'start': 0, 'end': 1, 'text': 'a'}])
    assert Model.__call__ is original
    assert profiler.profile.buckets['align_model.forward'].calls == 1
    profile = Profile({'total_align': _Bucket('total_align', 1., 1),
                       'align_model.forward': _Bucket('align_model.forward', .9, 1)})
    profile.report()
    assert '1.900' not in capsys.readouterr().out


def test_real_onnx_bundle_roundtrip(tmp_path):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxscript')
    pytest.importorskip('onnxruntime')
    from whisperx.alignment_engine import export_onnx_alignment

    class TinyCTC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(1, 3, 400, 320)
            self.projection = torch.nn.Linear(3, 3)

        def forward(self, waveforms, lengths=None):
            return self.projection(self.conv(waveforms[:, None]).transpose(1, 2)), (lengths-400)//320+1

    torch.manual_seed(8)
    model = TinyCTC().eval()
    metadata = {'language': 'en', 'dictionary': {'<pad>': 0, 'a': 1, 'b': 2}, 'type': 'torchaudio'}
    path = export_onnx_alignment(model, metadata, tmp_path/'bundle')
    onnx_model, onnx_metadata = load_align_model('en', 'cpu', model_name=path, engine='onnx')
    assert onnx_metadata['opset'] == 23
    for batch, samples in [(1, 400), (2, 16000), (3, 32003)]:
        waveforms = torch.randn(batch, samples)
        lengths = torch.full((batch,), samples)
        with torch.inference_mode():
            expected, expected_lengths = model(waveforms, lengths)
        actual, actual_lengths = onnx_model.infer(waveforms, lengths)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(actual_lengths, expected_lengths)
    quantized = export_onnx_alignment(model, metadata, tmp_path/'quantized', quantize=True)
    import onnx
    assert any(node.op_type == 'MatMulInteger' for node in onnx.load(str(quantized/'model.onnx')).graph.node)
    quantized_model, _ = load_align_model('en', 'cpu', model_name=quantized, engine='onnx')
    actual, _ = quantized_model.infer(waveforms, lengths)
    torch.testing.assert_close(actual, expected, atol=.03, rtol=.03)


def test_hf_onnx_export_preserves_variable_length_attention_masks(tmp_path):
    pytest.importorskip('onnx')
    pytest.importorskip('onnxscript')
    pytest.importorskip('onnxruntime')
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
    from whisperx.alignment_engine import export_onnx_alignment

    config = Wav2Vec2Config(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        conv_dim=(4,) * 7, feat_extract_norm='layer', num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2, vocab_size=4, mask_time_prob=0,
    )
    torch.manual_seed(23)
    model = Wav2Vec2ForCTC(config).eval()
    metadata = {'type': 'huggingface', 'language': 'en', 'dictionary': {'<pad>': 0, 'a': 1, 'b': 2, '|': 3}}
    path = export_onnx_alignment(model, metadata, tmp_path/'hf-bundle')
    engine, _ = load_align_model('en', 'cpu', model_name=path, engine='onnx', onnx_threads=1)
    for count, samples in [(1, 400), (3, 16003)]:
        waveform = torch.randn(count, samples)
        lengths = torch.tensor([max(400, samples // (i + 1)) for i in range(count)])
        with torch.inference_mode():
            expected, expected_lengths = get_emission_engine(model, metadata).infer(waveform, lengths)
        actual, actual_lengths = engine.infer(waveform, lengths)
        torch.testing.assert_close(actual_lengths, expected_lengths)
        for index, length in enumerate(expected_lengths):
            torch.testing.assert_close(actual[index, :length], expected[index, :length], atol=1e-5, rtol=1e-4)
