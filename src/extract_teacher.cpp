#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: extract_teacher TARGET.gguf PROMPTS.txt OUTPUT.bin\n";
        return 1;
    }
    try {
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) throw std::runtime_error("could not load teacher");
        auto params = llama_context_default_params();
        params.n_ctx = 512;
        params.n_batch = 256;
        params.n_ubatch = 256;
        params.n_threads = 8;
        params.n_threads_batch = 8;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), params), llama_free);
        if (!ctx) throw std::runtime_error("could not initialize teacher context");
        llama_set_embeddings_nextn(ctx.get(), true, false);
        const auto * vocab = llama_model_get_vocab(model.get());
        const int width = llama_model_n_embd(model.get());
        const int vocab_size = llama_vocab_n_tokens(vocab);
        std::ifstream prompts(argv[2]);
        std::ofstream output(argv[3], std::ios::binary);
        if (!prompts || !output) throw std::runtime_error("could not open input or output");
        auto write_int = [&](int32_t value) {
            output.write(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        write_int(width);
        std::string prompt;
        int completed = 0;
        while (std::getline(prompts, prompt)) {
            std::vector<llama_token> tokens(256);
            int count = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
            if (count <= 0 || count > 192) throw std::runtime_error("prompt does not fit extraction context");
            tokens.resize(count);
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            std::vector<float> hidden;
            auto decode = [&](llama_token * start, int length) {
                if (llama_decode(ctx.get(), llama_batch_get_one(start, length))) {
                    throw std::runtime_error("teacher decode failed");
                }
                const auto * rows = llama_get_embeddings_nextn(ctx.get());
                if (!rows) throw std::runtime_error("teacher did not expose nextn hidden states");
                hidden.insert(hidden.end(), rows, rows + length * width);
                const auto * logits = llama_get_logits_ith(ctx.get(), -1);
                if (!logits) throw std::runtime_error("missing teacher logits");
                return llama_token(std::max_element(logits, logits + vocab_size) - logits);
            };
            auto next = decode(tokens.data(), count);
            for (int step = 0; step < 40; ++step) {
                tokens.push_back(next);
                if (llama_vocab_is_eog(vocab, next) || step == 39) break;
                next = decode(&tokens.back(), 1);
            }
            if (hidden.size() != (tokens.size() - 1) * width) {
                throw std::runtime_error("token/hidden count mismatch");
            }
            write_int(tokens.size());
            write_int(count);
            output.write(reinterpret_cast<const char *>(tokens.data()), tokens.size() * sizeof(llama_token));
            output.write(reinterpret_cast<const char *>(hidden.data()), hidden.size() * sizeof(float));
            output.flush();
            if (!output) throw std::runtime_error("teacher cache write failed");
            std::cerr << "cached " << ++completed << " prompts\n";
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
