// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <filesystem>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/image_processor.hpp"

#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/unet2d_condition_model.hpp"

#include "openvino/runtime/core.hpp"

#include "json_utils.hpp"
#include "lora_helper.hpp"
#include "debug_utils.hpp"
#include "numpy_utils.hpp"

namespace ov {
namespace genai {

class StableDiffusionPipeline : public DiffusionPipeline {
public:
    explicit StableDiffusionPipeline(PipelineType pipeline_type) :
        DiffusionPipeline(pipeline_type) {
        // TODO: support GPU as well
        const std::string device = "CPU";

        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
            const bool do_normalize = true, do_binarize = false, gray_scale_source = false;
            m_image_processor = std::make_shared<ImageProcessor>(device, do_normalize, do_binarize, gray_scale_source);
            m_image_resizer = std::make_shared<ImageResizer>(device, ov::element::u8, "NHWC", ov::op::v11::Interpolate::InterpolateMode::BICUBIC_PILLOW);
        }

        if (m_pipeline_type == PipelineType::INPAINTING) {
            bool do_normalize = false, do_binarize = true;
            m_mask_processor_rgb = std::make_shared<ImageProcessor>(device, do_normalize, do_binarize, false);
            m_mask_processor_gray = std::make_shared<ImageProcessor>(device, do_normalize, do_binarize, true);
            m_mask_resizer = std::make_shared<ImageResizer>(device, ov::element::f32, "NCHW", ov::op::v11::Interpolate::InterpolateMode::NEAREST);
        }
    }

    StableDiffusionPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) :
        StableDiffusionPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet");
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder");
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    StableDiffusionPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties) :
        StableDiffusionPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        auto updated_properties = update_adapters_in_properties(properties, &DiffusionPipeline::derived_adapters);

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir / "text_encoder", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet", device, *updated_properties);
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, *updated_properties);
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder", device, *updated_properties);
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());

        update_adapters_from_properties(properties, m_generation_config.adapters);
    }

    StableDiffusionPipeline(
        PipelineType pipeline_type,
        const CLIPTextModel& clip_text_model,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae)
        : StableDiffusionPipeline(pipeline_type) {
        m_clip_text_encoder = std::make_shared<CLIPTextModel>(clip_text_model);
        m_unet = std::make_shared<UNet2DConditionModel>(unet);
        m_vae = std::make_shared<AutoencoderKL>(vae);

        const bool is_lcm = m_unet->get_config().time_cond_proj_dim > 0;
        const char * const pipeline_name = is_lcm ? "LatentConsistencyModelPipeline" : "StableDiffusionPipeline";
        initialize_generation_config(pipeline_name);
    }

    StableDiffusionPipeline(PipelineType pipeline_type, const StableDiffusionPipeline& pipe) :
        StableDiffusionPipeline(pipe) {
        OPENVINO_ASSERT(!pipe.is_inpainting_model(), "Cannot create ",
            pipeline_type == PipelineType::TEXT_2_IMAGE ? "'Text2ImagePipeline'" : "'Image2ImagePipeline'", " from InpaintingPipeline with inpainting model");

        m_pipeline_type = pipeline_type;

        const bool is_lcm = m_unet->get_config().time_cond_proj_dim > 0;
        const char * const pipeline_name = is_lcm ? "LatentConsistencyModelPipeline" : "StableDiffusionPipeline";
        initialize_generation_config(pipeline_name);
    }

    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) override {
        check_image_size(height, width);

        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        m_clip_text_encoder->reshape(batch_size_multiplier);
        m_unet->reshape(num_images_per_prompt * batch_size_multiplier, height, width, m_clip_text_encoder->get_config().max_position_embeddings);
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        compile(device, device, device, properties);
    }

    void compile(const std::string& text_encode_device,
        const std::string& denoise_device,
        const std::string& vae_decode_device,
        const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);
        auto updated_properties = update_adapters_in_properties(properties, &DiffusionPipeline::derived_adapters);

        m_clip_text_encoder->compile(text_encode_device, *updated_properties);
        m_unet->compile(denoise_device, *updated_properties);
        m_vae->compile(vae_decode_device, *updated_properties);
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        const auto& unet_config = m_unet->get_config();
        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG

        std::string negative_prompt = generation_config.negative_prompt != std::nullopt ? *generation_config.negative_prompt : std::string{};
        ov::Tensor encoder_hidden_states = m_clip_text_encoder->infer(positive_prompt, negative_prompt,
            batch_size_multiplier > 1);

        // replicate encoder hidden state to UNet model
        if (generation_config.num_images_per_prompt == 1) {
            // reuse output of text encoder directly w/o extra memory copy
            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states);
        } else {
            ov::Shape enc_shape = encoder_hidden_states.get_shape();
            enc_shape[0] *= generation_config.num_images_per_prompt;

            ov::Tensor encoder_hidden_states_repeated(encoder_hidden_states.get_element_type(), enc_shape);
            for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                numpy_utils::batch_copy(encoder_hidden_states, encoder_hidden_states_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    numpy_utils::batch_copy(encoder_hidden_states, encoder_hidden_states_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states_repeated);
        }

        if (unet_config.time_cond_proj_dim >= 0) { // LCM
            ov::Tensor timestep_cond = get_guidance_scale_embedding(generation_config.guidance_scale - 1.0f, unet_config.time_cond_proj_dim);
            m_unet->set_hidden_states("timestep_cond", timestep_cond);
        }
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) const override {
        std::vector<int64_t> timesteps = m_scheduler->get_timesteps();
        OPENVINO_ASSERT(!timesteps.empty(), "Timesteps are not computed yet");
        int64_t latent_timestep = timesteps.front();

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const bool is_inpainting = m_pipeline_type == PipelineType::INPAINTING,
            is_strength_max = is_inpainting && generation_config.strength == 1.0f,
            return_image_latent = is_inpainting && !is_inpainting_model();

        ov::Shape latent_shape{generation_config.num_images_per_prompt, m_vae->get_config().latent_channels,
                               generation_config.height / vae_scale_factor, generation_config.width / vae_scale_factor};
        ov::Tensor latent(ov::element::f32, {}), proccesed_image, image_latent, noise;

        if (initial_image) {
            proccesed_image = m_image_resizer->execute(initial_image, generation_config.height, generation_config.width);
            proccesed_image = m_image_processor->execute(proccesed_image);

            // prepate image latent for cases:
            // - image to image
            // - inpainting with strength < 1.0
            // - inpainting with non-specialized model
            if (!is_strength_max || return_image_latent) {
                image_latent = m_vae->encode(proccesed_image, generation_config.generator);

                // in case of image to image or inpaining with strength < 1.0, we need to initialize initial latent with image_latent
                if (!is_strength_max) {
                    image_latent.copy_to(latent);
                    latent = numpy_utils::repeat(latent, generation_config.num_images_per_prompt);
                }
            }
        }

        noise = generation_config.generator->randn_tensor(latent_shape);

        if (!latent.get_shape().empty()) {
            m_scheduler->add_noise(latent, noise, latent_timestep);
        } else {
            latent.set_shape(latent_shape);

            // if pure noise then scale the initial latents by the  Scheduler's init sigma
            const float * noise_data = noise.data<const float>();
            float * latent_data = latent.data<float>();
            for (size_t i = 0; i < latent.get_size(); ++i)
                latent_data[i] = noise_data[i] * m_scheduler->get_init_noise_sigma();
        }

        return std::make_tuple(latent, proccesed_image, image_latent, noise);
    }

    std::tuple<ov::Tensor, ov::Tensor> prepare_mask_latents(ov::Tensor mask_image, ov::Tensor processed_image, const ImageGenerationConfig& generation_config) {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::INPAINTING, "'prepare_mask_latents' can be called for inpainting pipeline only");

        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        ov::Shape target_shape = processed_image.get_shape();

        ov::Tensor mask_condition = m_image_resizer->execute(mask_image, target_shape[2], target_shape[3]);
        std::shared_ptr<IImageProcessor> mask_processor = mask_condition.get_shape()[3] == 1 ? m_mask_processor_gray : m_mask_processor_rgb;
        mask_condition = mask_processor->execute(mask_condition);

        // resize mask to shape of latent space
        ov::Tensor mask = m_mask_resizer->execute(mask_condition, target_shape[2] / vae_scale_factor, target_shape[3] / vae_scale_factor);
        mask = numpy_utils::repeat(mask, generation_config.num_images_per_prompt * batch_size_multiplier);

        ov::Tensor masked_image_latent;

        if (is_inpainting_model()) {
            // create masked image
            ov::Tensor masked_image(ov::element::f32, processed_image.get_shape());
            const float * mask_condition_data = mask_condition.data<const float>();
            const float * processed_image_data = processed_image.data<const float>();
            float * masked_image_data = masked_image.data<float>();

            for (size_t i = 0, plane_size = mask_condition.get_shape()[2] * mask_condition.get_shape()[3]; i < mask_condition.get_size(); ++i) {
                masked_image_data[i + 0 * plane_size] = mask_condition_data[i] < 0.5f ? processed_image_data[i + 0 * plane_size] : 0.0f;
                masked_image_data[i + 1 * plane_size] = mask_condition_data[i] < 0.5f ? processed_image_data[i + 1 * plane_size] : 0.0f;
                masked_image_data[i + 2 * plane_size] = mask_condition_data[i] < 0.5f ? processed_image_data[i + 2 * plane_size] : 0.0f;
            }

            // encode masked image to latent scape
            masked_image_latent = m_vae->encode(masked_image, generation_config.generator);
            masked_image_latent = numpy_utils::repeat(masked_image_latent, generation_config.num_images_per_prompt * batch_size_multiplier);
        }

        return std::make_tuple(mask, masked_image_latent);
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) override {
        if(adapters) {
            if(auto updated_adapters = derived_adapters(*adapters)) {
                adapters = updated_adapters;
            }
            m_clip_text_encoder->set_adapters(adapters);
            m_unet->set_adapters(adapters);
        }
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const ov::AnyMap& properties) override {
        using namespace numpy_utils;
        ImageGenerationConfig generation_config = m_generation_config;
        generation_config.update_generation_config(properties);

        // use callback if defined
        std::function<bool(size_t, size_t, ov::Tensor&)> callback = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback = callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>();
        }

        // Stable Diffusion pipeline
        // see https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline

        const auto& unet_config = m_unet->get_config();
        const size_t batch_size_multiplier = m_unet->do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        if (generation_config.height < 0)
            compute_dim(generation_config.height, initial_image, 1 /* assume NHWC */);
        if (generation_config.width < 0)
            compute_dim(generation_config.width, initial_image, 2 /* assume NHWC */);

        check_inputs(generation_config, initial_image);

        set_lora_adapters(generation_config.adapters);

        m_scheduler->set_timesteps(generation_config.num_inference_steps, generation_config.strength);
        std::vector<std::int64_t> timesteps = m_scheduler->get_timesteps();

        // compute text encoders and set hidden states
        compute_hidden_states(positive_prompt, generation_config);

        // preparate initial / image latents
        ov::Tensor latent, processed_image, image_latent, noise;
        std::tie(latent, processed_image, image_latent, noise) = prepare_latents(initial_image, generation_config);

        // prepare mask latents
        ov::Tensor mask, masked_image_latent;
        if (m_pipeline_type == PipelineType::INPAINTING) {
            std::tie(mask, masked_image_latent) = prepare_mask_latents(mask_image, processed_image, generation_config);
        }

        // prepare latents passed to models taking into account guidance scale (batch size multipler)
        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;

        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg), denoised, noisy_residual_tensor(ov::element::f32, {}), latent_model_input;

        for (size_t inference_step = 0; inference_step < timesteps.size(); inference_step++) {
            numpy_utils::batch_copy(latent, latent_cfg, 0, 0, generation_config.num_images_per_prompt);
            // concat the same latent twice along a batch dimension in case of CFG
            if (batch_size_multiplier > 1) {
                numpy_utils::batch_copy(latent, latent_cfg, 0, generation_config.num_images_per_prompt, generation_config.num_images_per_prompt);
            }

            m_scheduler->scale_model_input(latent_cfg, inference_step);

            ov::Tensor latent_model_input = is_inpainting_model() ? numpy_utils::concat(numpy_utils::concat(latent_cfg, mask, 1), masked_image_latent, 1) : latent_cfg;
            ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
            ov::Tensor noise_pred_tensor = m_unet->infer(latent_model_input, timestep);

            ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
            noise_pred_shape[0] /= batch_size_multiplier;

            if (batch_size_multiplier > 1) {
                noisy_residual_tensor.set_shape(noise_pred_shape);

                // perform guidance
                float* noisy_residual = noisy_residual_tensor.data<float>();
                const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
                const float* noise_pred_text = noise_pred_uncond + noisy_residual_tensor.get_size();

                for (size_t i = 0; i < noisy_residual_tensor.get_size(); ++i) {
                    noisy_residual[i] = noise_pred_uncond[i] +
                        generation_config.guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);
                }
            } else {
                noisy_residual_tensor = noise_pred_tensor;
            }

            auto scheduler_step_result = m_scheduler->step(noisy_residual_tensor, latent, inference_step, generation_config.generator);
            latent = scheduler_step_result["latent"];

            // in case of non-specialized inpainting model, we need manually mask current denoised latent and initial image latent
            if (m_pipeline_type == PipelineType::INPAINTING && !is_inpainting_model()) {
                blend_latents(image_latent, noise, mask, latent, inference_step);
            }

            // check whether scheduler returns "denoised" image, which should be passed to VAE decoder
            const auto it = scheduler_step_result.find("denoised");
            denoised = it != scheduler_step_result.end() ? it->second : latent;

            if (callback && callback(inference_step, timesteps.size(), denoised)) {
                return ov::Tensor(ov::element::u8, {});
            }
        }

        return decode(denoised);
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        return m_vae->decode(latent);
    }

protected:
    bool is_inpainting_model() const {
        assert(m_unet != nullptr);
        assert(m_vae != nullptr);
        return m_unet->get_config().in_channels == (m_vae->get_config().latent_channels * 2 + 1);
    }

    void compute_dim(int64_t & generation_config_value, ov::Tensor initial_image, int dim_idx) {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& unet_config = m_unet->get_config();

        // in case of image to image generation_config_value is just ignored and computed based on initial image
        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            OPENVINO_ASSERT(initial_image, "Initial image is empty for image to image pipeline");
            ov::Shape shape = initial_image.get_shape();
            int64_t dim_val = shape[dim_idx];

            generation_config_value = dim_val - (dim_val % vae_scale_factor);
        }

        if (generation_config_value < 0)
            generation_config_value = unet_config.sample_size * vae_scale_factor;
    }

    void initialize_generation_config(const std::string& class_name) override {
        assert(m_unet != nullptr);
        assert(m_vae != nullptr);
        const auto& unet_config = m_unet->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();

        // in case of image to image, the shape is computed based on initial image
        if (m_pipeline_type != PipelineType::IMAGE_2_IMAGE) {
            m_generation_config.height = unet_config.sample_size * vae_scale_factor;
            m_generation_config.width = unet_config.sample_size * vae_scale_factor;
        }

        if (class_name == "StableDiffusionPipeline" || class_name == "StableDiffusionImg2ImgPipeline" || class_name == "StableDiffusionInpaintPipeline") {
            m_generation_config.guidance_scale = 7.5f;
            m_generation_config.num_inference_steps = 50;
            m_generation_config.strength = m_pipeline_type == PipelineType::IMAGE_2_IMAGE ? 0.8f : 1.0f;
        } else if (class_name == "LatentConsistencyModelPipeline" || class_name == "LatentConsistencyModelImg2ImgPipeline") {
            m_generation_config.guidance_scale = 8.5f;
            m_generation_config.num_inference_steps = 4;
            m_generation_config.strength = m_pipeline_type == PipelineType::IMAGE_2_IMAGE ? 0.8f : 1.0f;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_vae != nullptr);
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) &&
            (width % vae_scale_factor == 0 || width < 0), "Both 'width' and 'height' must be divisible by ",
            vae_scale_factor);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.width, generation_config.height);

        const bool is_classifier_free_guidance = m_unet->do_classifier_free_guidance(generation_config.guidance_scale);
        const bool is_lcm = m_unet->get_config().time_cond_proj_dim > 0;
        const char * const pipeline_name = is_lcm ? "Latent Consistency Model" : "Stable Diffusion";

        OPENVINO_ASSERT(generation_config.prompt_2 == std::nullopt, "Prompt 2 is not used by ", pipeline_name);
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by ", pipeline_name);
        if (is_lcm) {
            OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used by ", pipeline_name);
        } else if (!is_classifier_free_guidance) {
            OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used when guidance scale <= 1.0");
        }
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by ", pipeline_name);
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by ", pipeline_name);

        if ((m_pipeline_type == PipelineType::IMAGE_2_IMAGE || m_pipeline_type == PipelineType::INPAINTING) && initial_image) {
            OPENVINO_ASSERT(generation_config.strength >= 0.0f && generation_config.strength <= 1.0f,
                "'Strength' generation parameter must be withion [0, 1] range");
        } else {
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
        }
    }

    friend class Text2ImagePipeline;
    friend class Image2ImagePipeline;

    std::shared_ptr<CLIPTextModel> m_clip_text_encoder = nullptr;
    std::shared_ptr<UNet2DConditionModel> m_unet = nullptr;
    std::shared_ptr<AutoencoderKL> m_vae = nullptr;
    std::shared_ptr<IImageProcessor> m_image_processor = nullptr, m_mask_processor_rgb = nullptr, m_mask_processor_gray = nullptr;
    std::shared_ptr<ImageResizer> m_image_resizer = nullptr, m_mask_resizer = nullptr;
};

}  // namespace genai
}  // namespace ov
