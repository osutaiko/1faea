import unittest

from peft import LoraConfig, get_peft_model
import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

from conversation_input_adapter import AdaptedConversationReader
from conversation_emoji_features import EmojiFeatureModel
from conversation_model import ConversationReader, memory
from conversation_data import IDS
from emoji_catalog import ALPHABET


class InputAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
        self.reader = AdaptedConversationReader.__new__(AdaptedConversationReader)
        self.reader.backbone = LlamaModel(config).eval().requires_grad_(False)
        self.reader.symbol_embedding = nn.Embedding(len(ALPHABET), 16).requires_grad_(False)

    def adapt(self):
        config = LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'],
                            layers_to_transform=[1], layers_pattern='layers', bias='none')
        self.reader.backbone = get_peft_model(self.reader.backbone, config).eval()
        for name, parameter in self.reader.backbone.named_parameters():
            if 'lora_B' in name:
                with torch.no_grad():
                    parameter.fill_(0.25)

    def test_input_adapter_cannot_change_emoji_memory_features(self):
        state, mask = memory([[]], [[IDS['💖'], IDS['🍕']]])
        original = ConversationReader.emoji_inputs(self.reader, state, mask)
        self.adapt()
        active = ConversationReader.emoji_inputs(self.reader, state, mask)
        disabled = self.reader.emoji_inputs(state, mask)
        self.assertGreater((active - original).abs().max().item(), 0.0001)
        torch.testing.assert_close(disabled, original)

    def test_only_final_layer_low_rank_parameters_receive_gradients(self):
        self.adapt()
        trainable = [name for name, parameter in self.reader.backbone.named_parameters() if parameter.requires_grad]
        self.assertEqual(len(trainable), 4)
        self.assertTrue(all('layers.1.' in name and 'lora_' in name for name in trainable))
        state, mask = memory([[]], [[IDS['💖'], IDS['🍕']]])
        embedding = self.reader.symbol_embedding(state)
        output = self.reader.backbone(inputs_embeds=embedding, attention_mask=mask, use_cache=False)
        output.last_hidden_state.square().mean().backward()
        for name, parameter in self.reader.backbone.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)
            else:
                self.assertIsNone(parameter.grad)


class EmojiFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = EmojiFeatureModel(torch.randn(len(ALPHABET), 16), width=16, layers=1).eval()

    def test_distilled_features_are_causal_and_padding_independent(self):
        state = torch.tensor([[IDS['💖'], IDS['🍕'], IDS['🧠']]])
        mask = torch.ones_like(state, dtype=torch.bool)
        original = self.model(state, mask)
        changed = state.clone()
        changed[0, -1] = IDS['🍣']
        torch.testing.assert_close(self.model(changed, mask)[:, :2], original[:, :2])
        padded = torch.cat((state, torch.tensor([[IDS['⚪'], IDS['⚪']]])), dim=1)
        padded_mask = torch.tensor([[True, True, True, False, False]])
        torch.testing.assert_close(self.model(padded, padded_mask)[:, :3], original)

    def test_distillation_trains_processor_without_changing_named_embeddings(self):
        state, mask = memory([[]], [[IDS['💖'], IDS['🍕']]])
        self.model(state, mask).square().mean().backward()
        self.assertIsNone(self.model.embedding.weight.grad)
        self.assertIsNotNone(self.model.output[-1].weight.grad)
        self.assertIsNotNone(self.model.projection[-1].weight.grad)
        with self.assertRaisesRegex(ValueError, 'integer'):
            self.model(state.float(), mask)


if __name__ == '__main__':
    unittest.main()
