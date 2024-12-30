#include "ModelLoader.h"

/**
 * @brief Constructor to create the model
 *        Note: Model must be in torchscript, cannot be pytorch model for inferencing
 * @param model_path Path to torch script model to be loaded
 */
torch_model::torch_model(const std::string& model_path) {
    try {
        if (!model_path) {
            std::cout << "No model found\n";
            return -1;
        }

        torch::jit::script::Module module;

        try {
            this.model = torch::jot::load(model_path);
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model\n";
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error instantiating model\n";
        return -1;
    }

    std::cout << "Model loaded succesfully\n";
}

torch_model::inference(const std::string& message) {

}

void torch_model::encoder() {

}