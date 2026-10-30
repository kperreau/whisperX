"""Offline CPU int8 round trips using small, randomly initialized wav2vec2 models."""

from __future__ import annotations

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torchaudio

from whisperx import alignment_native as align_cache
from whisperx.alignment import load_align_model
from whisperx.alignment_engine import get_emission_engine, infer_emissions


class AlignCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.path = str(Path(cls.directory.name, "en.int8.pt"))
        with torch.random.fork_rng():
            torch.manual_seed(0)
            cls.eager = torchaudio.models.wav2vec2_model(
                extractor_mode="group_norm",
                extractor_conv_layer_config=[
                    (4, 10, 5), (4, 3, 2), (4, 3, 2), (4, 3, 2),
                    (4, 3, 2), (4, 2, 2), (4, 2, 2),
                ],
                extractor_conv_bias=False, encoder_embed_dim=8,
                encoder_projection_dropout=0, encoder_pos_conv_kernel=4,
                encoder_pos_conv_groups=2, encoder_num_layers=1, encoder_num_heads=2,
                encoder_attention_dropout=0, encoder_ff_interm_features=16,
                encoder_ff_interm_dropout=0, encoder_dropout=0,
                encoder_layer_norm_first=False, encoder_layer_drop=0, aux_num_out=4,
            ).eval()
        align_cache._bake_cache(cls.eager, "torchaudio", cls.path)

    def test_cache_preserves_dynamic_batches_lengths_and_ctc_posteriors(self):
        align_cache._verify_cache(self.eager, self.path)

    def test_cache_restores_structure_without_reinitializing_the_model(self):
        with mock.patch.object(torchaudio.models, "wav2vec2_model", side_effect=AssertionError("model reconstruction")):
            cached = align_cache._CachedModel(self.path)
            emissions, lengths = cached.infer(torch.zeros(1, 8000), torch.tensor([8000]))
            self.assertEqual(emissions.shape, (1, 24, 4))
            self.assertEqual(lengths.tolist(), [24])

    def test_sensitive_encoder_block_can_remain_float_without_reconstructing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "mixed.pt"))
            align_cache._bake_cache(self.eager, "torchaudio", path,
                                      float_prefixes=("encoder.transformer.layers.0",))
            cached = align_cache._CachedModel(path)
            self.assertIsInstance(cached.model.encoder.transformer.layers[0].attention.q_proj, torch.nn.Linear)
            self.assertIsInstance(cached.model.encoder.feature_projection.projection, align_cache.Int8Linear)

    def test_cache_uses_regular_integer_weights_and_weights_only_loading(self):
        with mock.patch.object(torch, "load", wraps=torch.load) as load:
            cached = align_cache._CachedModel(self.path, cpu_threads=1)
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertTrue(load.call_args.kwargs["mmap"])
        layers = [layer for layer in cached.model.modules() if isinstance(layer, align_cache.Int8Linear)]
        self.assertGreater(len(layers), 0)
        self.assertTrue(all(layer.qweight.dtype == torch.int8 for layer in layers))
        self.assertFalse(any(value.is_quantized for value in cached.model.state_dict().values()))

    def test_public_bundle_loader_runs_full_context_without_legacy_apis_or_providers(self):
        metadata = {"language": "en", "type": "torchaudio", "model_name": "fixture",
                    "dictionary": {"-": 0, "a": 1, "b": 2, "|": 3}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "bundle")
            with (
                mock.patch.object(torch.jit, "load", side_effect=AssertionError("legacy load")),
                mock.patch.object(torch, "quantize_per_tensor", side_effect=AssertionError("legacy quantizer")),
                mock.patch.object(torch, "quantize_per_channel", side_effect=AssertionError("legacy quantizer")),
                mock.patch("whisperx.alignment._load_torchaudio", side_effect=AssertionError("provider loader")),
            ):
                align_cache.export_native_alignment(self.eager, metadata, path)
                engine, actual_metadata = load_align_model(
                    "en", "cpu", model_name=path, engine="native-int8", cpu_threads=1,
                )
                self.assertIs(get_emission_engine(engine, actual_metadata), engine)
                self.assertEqual(actual_metadata["model_type"], "torchaudio")
                self.assertEqual(actual_metadata["quantization"],
                                 "native-int8-per-channel-weights-7bit-per-row-activations")
                self.assertEqual(actual_metadata["activation_levels"], 127)
                with mock.patch.object(engine, "infer", wraps=engine.infer) as run:
                    result = infer_emissions([torch.zeros(8000)], engine, "cpu", chunk_seconds=None)
                    self.assertEqual(result[0].shape, (24, 4))
                    run.assert_called_once()
                    self.assertEqual(run.call_args.args[0].shape, (1, 8000))
            with self.assertRaises(FileExistsError):
                align_cache.export_native_alignment(self.eager, metadata, path)
            with self.assertRaisesRegex(ValueError, "language"):
                load_align_model("fr", "cpu", model_name=path, engine="native-int8")
            with self.assertRaisesRegex(ValueError, "cpu"):
                load_align_model("en", "cuda", model_name=path, engine="native-int8")
            with self.assertRaisesRegex(ValueError, "local bundle"):
                load_align_model("en", "cpu", engine="native-int8")
            description = json.loads((path / "metadata.json").read_text())
            for version in (1, 99):
                description["format_version"] = version
                (path / "metadata.json").write_text(json.dumps(description))
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "re-export"):
                    align_cache.NativeEmissionEngine(path)

    def test_fp32_bundle_reproduces_the_source_model_bitwise_without_reconstruction(self):
        metadata = {"language": "en", "type": "torchaudio", "model_name": "WAV2VEC2_ASR_BASE_960H",
                    "dictionary": {"-": 0, "a": 1, "b": 2, "|": 3}}
        eager = copy.deepcopy(self.eager)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "fp32")
            align_cache.export_native_alignment(eager, metadata, path, precision="fp32")
            self.assertEqual(sorted(entry.name for entry in path.iterdir()), ["metadata.json", "model.fp32.pt"])
            manifest = json.loads((path / "metadata.json").read_text())
            self.assertEqual((manifest["type"], manifest["model_type"]), ("native-fp32", "torchaudio"))
            self.assertNotIn("quantization", manifest)
            with (
                mock.patch.object(torchaudio.models, "wav2vec2_model", side_effect=AssertionError("model reconstruction")),
                mock.patch.object(align_cache, "native_int8_supported", side_effect=AssertionError("int8 probe")),
            ):
                engine, _ = load_align_model("en", "cpu", model_name=path, engine="native-fp32", cpu_threads=1)
            self.assertFalse(any(isinstance(layer, align_cache.Int8Linear) for layer in engine.model.modules()))
            self.assertIsNone(engine.integer_fallback)
            waveforms, lengths = torch.randn(2, 16000, generator=torch.Generator().manual_seed(1)), torch.tensor([16000, 9000])
            with torch.inference_mode():
                expected = eager(waveforms, lengths)
                actual = engine.infer(waveforms, lengths)
            for left, right in zip(actual, expected, strict=True):
                self.assertTrue(torch.equal(left, right))
            with self.assertRaisesRegex(ValueError, "does not match engine"):
                load_align_model("en", "cpu", model_name=path, engine="native-int8")
            with self.assertRaisesRegex(RuntimeError, "precision mismatch"):
                align_cache._CachedModel(path / "model.fp32.pt", precision="int8")
            with self.assertRaisesRegex(ValueError, "float_prefixes"):
                align_cache.export_native_alignment(eager, metadata, Path(directory, "bad"), precision="fp32",
                                                    float_prefixes=())
            with self.assertRaisesRegex(ValueError, "precision"):
                align_cache.export_native_alignment(eager, metadata, Path(directory, "bad"), precision="fp16")

    def test_fp32_verification_rejects_any_difference(self):
        waveforms, lengths = torch.zeros(1, 400), torch.tensor([400])
        with torch.inference_mode():
            expected, output_lengths = self.eager(waveforms, lengths)
        with (
            mock.patch.object(align_cache, "_verification_inputs", return_value=[(waveforms, lengths)]),
            mock.patch.object(align_cache, "_CachedModel") as loader,
        ):
            loader.return_value.integer_fallback = None
            loader.return_value.infer.return_value = (expected + 1e-6, output_lengths)
            with self.assertRaisesRegex(RuntimeError, "differs from the source"):
                align_cache._verify_cache(self.eager, "unused", precision="fp32")

    def test_known_english_checkpoint_profile_and_invalid_exports(self):
        metadata = {"language": "en", "type": "torchaudio", "model_name": "WAV2VEC2_ASR_BASE_960H"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "english")
            align_cache.export_native_alignment(self.eager, metadata, path)
            engine = align_cache.NativeEmissionEngine(path)
            self.assertIsInstance(engine.model.encoder.transformer.layers[0].attention.q_proj, torch.nn.Linear)
            self.assertEqual(engine.metadata["float_prefixes"], ["encoder.transformer.layers.0"])
            with self.assertRaisesRegex(ValueError, "prefix"):
                align_cache.export_native_alignment(self.eager, metadata, Path(directory, "bad"),
                                                    float_prefixes=("does_not_exist",))
            with self.assertRaisesRegex(ValueError, "Torch"):
                align_cache.export_native_alignment(self.eager, {"type": "custom"}, Path(directory, "bad"))

    def test_known_french_checkpoint_preserves_sensitive_layers(self):
        metadata = {"language": "fr", "type": "torchaudio", "model_name": "VOXPOPULI_ASR_BASE_10K_FR"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "french")
            eager = copy.deepcopy(self.eager)
            eager.encoder.transformer.layers.extend(
                copy.deepcopy(eager.encoder.transformer.layers[0]) for _ in range(5)
            )
            align_cache.export_native_alignment(eager, metadata, path)
            # Both kernels compute the same integers, so they share one profile.
            for fallback in (False, True):
                engine = align_cache._CachedModel(path / "model.int8.pt", integer_fallback=fallback)
                self.assertIsInstance(engine.model.encoder.feature_projection.projection, torch.nn.Linear)
                self.assertIsInstance(engine.model.encoder.transformer.layers[0].attention.q_proj, align_cache.Int8Linear)
                self.assertIsInstance(engine.model.encoder.transformer.layers[5].attention.q_proj, torch.nn.Linear)
                self.assertEqual(engine.integer_fallback, fallback)
            with mock.patch.object(align_cache, "native_int8_supported", return_value=False):
                engine = align_cache.NativeEmissionEngine(path)
                self.assertEqual(engine.metadata["float_prefixes"],
                                 ["encoder.feature_projection.projection", "encoder.transformer.layers.5"])
            manifest = json.loads((path / "metadata.json").read_text())
            self.assertNotIn("integer_fallback_float_prefixes", manifest)


    def test_short_segment_preserves_zero_output_length_after_padding(self):
        cached = align_cache._CachedModel(self.path, cpu_threads=1)
        _, lengths = cached.infer(torch.zeros(2, 400), torch.tensor([1, 0]))
        self.assertEqual(lengths.tolist(), [0, 0])

    def test_later_inference_does_not_overwrite_retained_outputs(self):
        cached = align_cache._CachedModel(self.path, cpu_threads=1)
        first = cached.infer(torch.zeros(1, 8000), torch.tensor([8000]))
        saved = tuple(value.clone() for value in first)
        cached.infer(torch.ones(2, 16000), torch.tensor([16000, 8000]))
        for actual, expected in zip(first, saved, strict=True):
            torch.testing.assert_close(actual, expected)

    def test_verification_rejects_nonfinite_or_divergent_ctc_emissions(self):
        waveforms, lengths = torch.zeros(1, 400), torch.tensor([400])
        with torch.inference_mode():
            expected, output_lengths = self.eager(waveforms, lengths)
        for emissions, error in (
            (torch.full_like(expected, float("nan")), "invalid quantized"),
            (torch.arange(4).expand_as(expected) * 1000.0, "posterior divergence"),
        ):
            with (
                self.subTest(error=error),
                mock.patch.object(align_cache, "_verification_inputs", return_value=[(waveforms, lengths)]),
                mock.patch.object(align_cache, "_CachedModel") as loader,
            ):
                loader.return_value.infer.return_value = (emissions, output_lengths)
                with self.assertRaisesRegex(RuntimeError, error):
                    align_cache._verify_cache(self.eager, "unused")

    def test_verification_rejects_invalid_reference_and_nonfinite_divergence(self):
        waveforms, lengths = torch.zeros(1, 400), torch.tensor([400])
        output_lengths = torch.tensor([1])
        finite = torch.zeros(1, 1, 4)
        extreme = torch.tensor([[[torch.finfo(torch.float32).max,
                                  -torch.finfo(torch.float32).max, 0., 0.]]])
        for expected, actual, error in (
            (torch.full_like(finite, float("nan")), finite, "invalid reference"),
            (torch.full_like(finite, float("inf")), finite, "invalid reference"),
            (extreme, extreme, "posterior divergence"),
        ):
            with (
                self.subTest(error=error, reference=expected),
                mock.patch.object(align_cache, "_verification_inputs", return_value=[(waveforms, lengths)]),
                mock.patch.object(align_cache, "_CachedModel") as loader,
            ):
                reference = mock.Mock(model_type="torchaudio", return_value=(expected, output_lengths))
                loader.return_value.infer.return_value = (actual, output_lengths)
                with self.assertRaisesRegex(RuntimeError, error):
                    align_cache._verify_cache(reference, "unused")



    def test_invalid_cache_format_and_empty_quantized_state_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "invalid.pt"))
            version = align_cache.CACHE_FORMAT_VERSION
            for payload, error in (({"format": 99}, "cache format"),
                                   ({"format": 1, "model_type": "torchaudio"}, "re-export"),
                                   ({"format": version, "model_type": "torchaudio", "model": torch.nn.Identity()},
                                    "no quantized linear")):
                torch.save(payload, path)
                with self.assertRaisesRegex(RuntimeError, error):
                    align_cache._CachedModel(path)
        with self.assertRaisesRegex(RuntimeError, "unsupported alignment model"):
            align_cache.cache_globals("unknown")

    def test_cache_rejects_unlisted_pickle_globals_and_mismatched_model_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "invalid.pt"))
            torch.save({"model": Path("unused")}, path)
            with self.assertRaises(pickle.UnpicklingError):
                align_cache._CachedModel(path)
            torch.save({"format": align_cache.CACHE_FORMAT_VERSION, "model_type": "huggingface"}, path)
            with self.assertRaisesRegex(RuntimeError, "model type mismatch"):
                align_cache._CachedModel(path)

    def test_huggingface_preserves_mask_geometry_nondefault_normalization_and_bias(self):
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

        config = Wav2Vec2Config(
            hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, vocab_size=4, conv_dim=(4, 4, 4, 4, 4, 4, 4),
            conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=2,
            layer_norm_eps=1e-4, feat_extract_activation="silu", conv_bias=True,
        )
        with torch.random.fork_rng():
            torch.manual_seed(0)
            eager = Wav2Vec2ForCTC(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "hf.int8.pt"))
            align_cache._bake_cache(eager, "huggingface", path)


class Int8LinearTest(unittest.TestCase):
    def test_kernel_probe_caches_fallback_when_native_operators_are_unavailable(self):
        self.addCleanup(align_cache.native_int8_supported.cache_clear)
        for stage in ("pack", "run"):
            for error in (RuntimeError("unsupported CPU"), AttributeError("missing operator")):
                align_cache.native_int8_supported.cache_clear()
                with (
                    self.subTest(stage=stage, error=type(error).__name__),
                    mock.patch.object(torch.backends.mkldnn, "is_available", return_value=True),
                    mock.patch.object(torch.ops.onednn, "qlinear_prepack",
                                      side_effect=error if stage == "pack" else None,
                                      return_value=torch.empty(0)) as pack,
                    mock.patch.object(torch.ops.onednn.qlinear_pointwise, "tensor",
                                      side_effect=error if stage == "run" else None) as run,
                ):
                    self.assertFalse(align_cache.native_int8_supported())
                    self.assertFalse(align_cache.native_int8_supported())
                    self.assertEqual(pack.call_count, 1)
                    self.assertEqual(run.call_count, int(stage == "run"))

    def test_kernel_probe_is_cached_and_rejects_reduced_weight_arithmetic(self):
        align_cache.native_int8_supported.cache_clear()
        self.addCleanup(align_cache.native_int8_supported.cache_clear)
        # The largest 7-bit code times the extreme weights, summed over 256 inputs.
        expected = torch.tensor([[127. * 256 * 127] * 64 + [127. * 256 * -128] * 64] * 16)
        with (
            mock.patch.object(torch.backends.mkldnn, "is_available", return_value=True),
            mock.patch.object(torch.ops.onednn, "qlinear_prepack", return_value=torch.empty(0)) as pack,
            mock.patch.object(torch.ops.onednn.qlinear_pointwise, "tensor", return_value=expected) as run,
        ):
            self.assertTrue(align_cache.native_int8_supported())
            self.assertTrue(align_cache.native_int8_supported())
            self.assertEqual(pack.call_count, 1)
            self.assertEqual(run.call_count, 1)
            align_cache.native_int8_supported.cache_clear()
            run.return_value = expected / 2
            self.assertFalse(align_cache.native_int8_supported())
        align_cache.native_int8_supported.cache_clear()
        with mock.patch.object(torch.backends.mkldnn, "is_available", return_value=False):
            self.assertFalse(align_cache.native_int8_supported())

    def test_integer_fallback_matches_dequantized_reference_for_signs_zero_and_bias(self):
        for bias in (False, True):
            layer = align_cache.Int8Linear(8, 4, bias)
            layer.qweight.copy_(torch.arange(-16, 16, dtype=torch.int8).reshape(4, 8))
            layer.weight_scale.fill_(0.05)
            if bias:
                layer.bias.copy_(torch.arange(4, dtype=torch.float32))
            with mock.patch.object(align_cache, "native_int8_supported", return_value=False):
                layer.prepare()
            signed = torch.tensor([[-127., -64., 0., 63., 127., 0., -64., 64.]])
            for values in (torch.zeros(2, 8), torch.ones(2, 8), signed.repeat(2, 1),
                           torch.cat([signed, 1000 * signed, -signed.abs(), signed.abs()])):
                codes, scale, zero = align_cache.quantize_rows_reference(values)
                dequantized = (codes.float() - zero[:, None]) * scale[:, None]
                actual = layer(values)
                expected = dequantized @ (layer.qweight.float() * layer.weight_scale[:, None]).T
                if bias:
                    expected += layer.bias
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)

    def test_onednn_receives_ordinary_integer_tensors(self):
        layer = align_cache.Int8Linear(8, 4, False)
        layer.qweight.zero_()
        layer.weight_scale.fill_(1)
        with (
            mock.patch.object(align_cache, "native_int8_supported", return_value=True),
            mock.patch.object(torch.ops.onednn, "qlinear_prepack", return_value=layer.qweight),
            mock.patch.object(torch.ops.onednn.qlinear_pointwise, "tensor", return_value=torch.zeros(2, 4)) as run,
        ):
            layer.prepare()
            self.assertEqual(layer(torch.randn(2, 8) * 100).shape, (2, 4))
            codes, scale, zero = run.call_args.args[:3]
            self.assertEqual(codes.dtype, torch.uint8)
            self.assertFalse(codes.is_quantized)
            self.assertLessEqual(int(codes.max()), align_cache.ACTIVATION_LEVELS)
            # Per-row scales and zero points are applied after the GEMM.
            self.assertEqual((scale.item(), zero.item()), (1.0, 0))


class RowQuantizationTest(unittest.TestCase):
    def test_codes_fit_avx2_pair_sums_without_vnni(self):
        # vpmaddubsw adds two u8 x s8 products in a signed 16-bit lane.
        largest = 2 * align_cache.ACTIVATION_LEVELS * 128
        self.assertLess(largest, 2 ** 15)
        values = torch.randn(64, 96, generator=torch.Generator().manual_seed(0)) * 50
        codes, _, _ = align_cache.quantize_rows_reference(values)
        self.assertLessEqual(int(codes.max()), align_cache.ACTIVATION_LEVELS)

    def test_rows_are_quantized_independently(self):
        quiet = torch.linspace(-1, 1, 16)[None]
        alone, _, _ = align_cache.quantize_rows_reference(quiet)
        together, _, _ = align_cache.quantize_rows_reference(torch.cat([quiet, 1000 * quiet]))
        torch.testing.assert_close(together[:1], alone, rtol=0, atol=0)
        self.assertEqual(len(torch.unique(alone)), 16)

    def test_reference_rejects_nonfinite_rows(self):
        values = torch.tensor([[1., float("inf")], [1., 2.]])
        _, scale, _ = align_cache.quantize_rows_reference(values)
        self.assertTrue(scale[0].isnan())
        self.assertTrue(scale[1].isfinite())

    @unittest.skipIf(align_cache._quantize_rows is None, "whisperx_ext is not built")
    def test_rust_kernels_match_the_torch_reference_bit_for_bit(self):
        generator = torch.Generator().manual_seed(0)
        for rows, width, threads in ((1, 1, 1), (3, 7, 1), (70, 768, 2), (300, 9, 3), (129, 3072, 2)):
            values = torch.randn(rows, width, generator=generator) * torch.rand(rows, 1, generator=generator) * 10
            values[0] = values[0].abs()
            if rows > 2:
                values[1] = -values[1].abs()
                values[2, 0] = float("nan")
            codes, scale, zero = torch.empty(rows, width, dtype=torch.uint8), torch.empty(rows), torch.empty(rows)
            align_cache._quantize_rows(values.numpy(), codes.numpy(), scale.numpy(), zero.numpy(), threads)
            expected = align_cache.quantize_rows_reference(values)
            with self.subTest(rows=rows, width=width, threads=threads):
                self.assertTrue(torch.equal(codes, expected[0]))
                for actual, reference in ((scale, expected[1]), (zero, expected[2])):
                    self.assertTrue(torch.equal(actual.nan_to_num(-1), reference.nan_to_num(-1)))
                products = torch.randn(rows, 5, generator=generator)
                sums, bias = torch.randn(5, generator=generator), torch.randn(5, generator=generator)
                for offset in (None, bias):
                    actual = products.clone()
                    align_cache._dequantize_rows(actual.numpy(), scale.numpy(), zero.numpy(), sums.numpy(),
                                                 None if offset is None else offset.numpy(), threads)
                    reference = align_cache.dequantize_rows_reference(products.clone(), scale, zero, sums, offset)
                    self.assertTrue(torch.equal(actual.nan_to_num(-1), reference.nan_to_num(-1)))

    @unittest.skipIf(align_cache._quantize_rows is None, "whisperx_ext is not built")
    def test_rust_kernels_reject_mismatched_buffers(self):
        values = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "match"):
            align_cache._quantize_rows(values.numpy(), torch.empty(2, 4, dtype=torch.uint8).numpy(),
                                       torch.empty(2).numpy(), torch.empty(2).numpy())
        with self.assertRaisesRegex(ValueError, "match"):
            align_cache._dequantize_rows(values.numpy(), torch.empty(2).numpy(), torch.empty(2).numpy(),
                                         torch.empty(4).numpy())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            align_cache._quantize_rows(torch.zeros(3, 2).T.numpy(), torch.empty(2, 3, dtype=torch.uint8).numpy(),
                                       torch.empty(2).numpy(), torch.empty(2).numpy())


