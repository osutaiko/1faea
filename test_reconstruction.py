import unittest

import torch
from torch.nn import functional as F

from model import EMOJIS
from reconstruction import BOS, TextAutoencoder, batch_text, passages


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TextAutoencoder(4, width=16)
        self.source, self.prefix, self.target = batch_text(["A short passage.", "Another text."], "cpu")

    def test_discrete_training_and_inference_states_are_exactly_one_hot(self):
        for training in (True, False):
            self.model.train(training)
            state = self.model.encode(self.source)
            expected = F.one_hot(state.argmax(-1), len(EMOJIS)).float()
            self.assertTrue(torch.equal(state, expected))

    def test_decoder_receives_only_reembedded_codes(self):
        self.model.eval()
        state = self.model.encode(self.source)
        received = []
        handle = self.model.decoder.register_forward_pre_hook(
            lambda module, args: received.append(args[1])
        )
        self.model.decode(state, self.prefix)
        handle.remove()
        expected = self.model.codebook(state.argmax(-1)) + self.model.memory_position
        torch.testing.assert_close(received[0], expected, rtol=0, atol=0)

    def test_encoding_does_not_depend_on_other_passage_lengths(self):
        self.model.eval()
        alone, _, _ = batch_text(["Short text."], "cpu")
        together, _, _ = batch_text(["Short text.", "A much longer passage for padding."], "cpu")
        self.assertTrue(torch.equal(self.model.encode(alone)[0], self.model.encode(together)[0]))

    def test_scalar_states_are_exact_code_lookups_in_training_and_inference(self):
        model = TextAutoencoder(4, width=16, quantization="scalar")
        for training in (True, False):
            model.train(training)
            state = model.encode(self.source)
            indices = model.state_indices(state)
            self.assertTrue(((indices >= 0) & (indices < 64)).all())
            expected = model.scalar_quantizer.indices_to_codes(indices)
            torch.testing.assert_close(state, expected, rtol=0, atol=0)

    def test_scalar_decoder_memory_depends_only_on_selected_codes(self):
        model = TextAutoencoder(4, width=16, quantization="scalar").eval()
        state = model.encode(self.source)
        ids = model.state_indices(state)
        recovered = model.scalar_quantizer.indices_to_codes(ids)
        torch.testing.assert_close(model.decode(state, self.prefix),
                                   model.decode(recovered, self.prefix), rtol=0, atol=0)

    def test_scalar_reconstruction_gradient_reaches_the_encoder(self):
        model = TextAutoencoder(4, width=16, quantization="scalar")
        logits = model(self.source, self.prefix)
        F.cross_entropy(logits.flatten(0, 1), self.target.flatten()).backward()
        self.assertGreater(model.code_head[1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder.layers[0].linear1.weight.grad.abs().sum().item(), 0)

    def test_future_target_bytes_do_not_affect_earlier_predictions(self):
        self.model.eval()
        state = self.model.encode(self.source)
        prefix = torch.tensor([[BOS, 65, 66], [BOS, 67, 68]])
        changed = prefix.clone()
        changed[:, 2] = 69
        first = self.model.decode(state, prefix)
        second = self.model.decode(state, changed)
        torch.testing.assert_close(first[:, :2], second[:, :2], rtol=0, atol=0)

    def test_reconstruction_loss_reaches_source_encoder_through_codes(self):
        logits = self.model(self.source, self.prefix)
        F.cross_entropy(logits.flatten(0, 1), self.target.flatten()).backward()
        for parameter in (self.model.code_head[1].weight, self.model.encoder.layers[0].linear1.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_baseline_has_identical_parameters_but_continuous_states(self):
        baseline = TextAutoencoder(4, discrete=False, width=16)
        baseline.load_state_dict(self.model.state_dict())
        state = baseline.encode(self.source)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()),
                         sum(p.numel() for p in self.model.parameters()))
        self.assertTrue(((state > 0) & (state < 1)).all())

    def test_utf8_targets_are_the_original_bytes_plus_end_token(self):
        source, prefix, target = batch_text(["café"], "cpu")
        original = list("café".encode("utf-8"))
        self.assertEqual(source[0].tolist(), original)
        self.assertEqual(prefix[0, 1:].tolist(), original)
        self.assertEqual(target[0, :-1].tolist(), original)
        self.assertEqual(target[0, -1].item(), 256)

    def test_long_unicode_words_are_split_without_breaking_utf8(self):
        original = "“important—unimportant—unimportant—important—”" * 2
        chunks = list(passages(original, 48))
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 48 for chunk in chunks))
        self.assertEqual("".join(chunks), original)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
