from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    """
    LLaVA 模型的配置类，继承自 Hugging Face 的 LlamaConfig 类。

    LLaVA（Large Language and Vision Assistant）模型是一个多模态模型，结合了语言和视觉处理能力。
    该配置类扩展了原始的 Llama 模型配置，添加了特定于 LLaVA 的参数和设置。
    """
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    """
    初始化 LLaVA 配置类。

    参数:
        **kwargs: 其他关键字参数，会传递给 LlamaConfig 的初始化方法。
    """
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    """
    LLaVA 因果语言模型类，继承自 Hugging Face 的 LlamaForCausalLM 和 LLaVA 的元因果语言模型。

    该类结合了 LLaVA 的多模态处理能力和原始的 Llama 因果语言模型功能，实现了多模态因果语言建模。
    """
    # 指定配置类为 LlavaConfig
    config_class = LlavaConfig

    def __init__(self, config):
        """
        初始化 LLaVA 因果语言模型。

        参数:
            config: 模型配置，包含模型的各个参数设置。
        """
        super(LlamaForCausalLM, self).__init__(config)
        # 初始化 LLaVA 模型的实例
        self.model = LlavaLlamaModel(config)
        # 获取预训练时的张量并行度
        self.pretraining_tp = config.pretraining_tp
        # 获取词汇表大小
        self.vocab_size = config.vocab_size
        # 初始化语言模型头（线性层）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_model(self):
        """
        获取 LLaVA 模型的实例。

        返回:
            LlavaLlamaModel: LLaVA 模型的实例。
        """
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播函数，实现 LLaVA 因果语言模型的前向计算。

        参数:
            input_ids (torch.LongTensor, 可选): 输入的文本标记 ID。
            attention_mask (Optional[torch.Tensor], 可选): 注意力掩码。
            position_ids (Optional[torch.LongTensor], 可选): 位置 ID。
            past_key_values (Optional[List[torch.FloatTensor]], 可选): 过去的键值对。
            inputs_embeds (Optional[torch.FloatTensor], 可选): 输入嵌入。
            labels (Optional[torch.LongTensor], 可选): 标签。
            use_cache (Optional[bool], 可选): 是否使用缓存。
            output_attentions (Optional[bool], 可选): 是否输出注意力权重。
            output_hidden_states (Optional[bool], 可选): 是否输出隐藏状态。
            images (Optional[torch.FloatTensor], 可选): 输入的图像张量。
            image_sizes (Optional[List[List[int]]], 可选): 图像大小列表。
            return_dict (Optional[bool], 可选): 是否返回字典格式的输出。

        返回:
            Union[Tuple, CausalLMOutputWithPast]: 模型输出。
        """

        if inputs_embeds is None:
            # 如果没有提供输入嵌入，则准备输入和标签
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        # 调用父类的前向方法进行前向传播
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        生成文本。

        参数:
            inputs (Optional[torch.Tensor], 可选): 输入张量。
            images (Optional[torch.Tensor], 可选): 输入图像张量。
            image_sizes (Optional[torch.Tensor], 可选): 图像大小张量。
            **kwargs: 其他关键字参数。

        返回:
            Union[GenerateOutput, torch.LongTensor]: 生成结果。
        """
        # 获取位置 ID
        position_ids = kwargs.pop("position_ids", None)
        # 获取注意力掩码
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            # 如果提供了输入嵌入，则抛出未实现错误
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            # 如果提供了图像，则准备输入和标签
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            # 否则，使用嵌入层获取输入嵌入
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 调用父类的生成方法进行文本生成
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        """
        准备生成时的输入。

        参数:
            input_ids: 输入的文本标记 ID。
            past_key_values (Optional, 可选): 过去的键值对。
            inputs_embeds (Optional, 可选): 输入嵌入。
            **kwargs: 其他关键字参数。

        返回:
            dict: 准备好的输入。
        """
        # 获取图像
        images = kwargs.pop("images", None)
        # 获取图像大小
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            # 添加图像到输入字典
            inputs['images'] = images
        if image_sizes is not None:
            # 添加图像大小到输入字典
            inputs['image_sizes'] = image_sizes
        return inputs

# 注册配置和模型类
AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
