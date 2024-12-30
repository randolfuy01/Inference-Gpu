#ifndef MODELLOADER_H
#define MODELLOADER_H

#include <torch/script.h> 
#include <torch/torch.h> 
#include <iostream>
#include <memory>

class torch_model {
    public:
        std::shared_ptr<torch::jit::script::Module> model;
        explicit torch_model(const std::string& model_path);
        int inference(const std::string& message);
        void encoder();
};

#endif
